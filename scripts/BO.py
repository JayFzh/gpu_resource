from bayes_opt import BayesianOptimization
import os
import numpy as np

# opt setting
init_num = 10
iter_num = 100
tmp_file = "train_log"
log_file = "alloc_log"
log = open(log_file, "w")
group_size = 80
total_layer = 622
group_num = int(np.ceil(total_layer / group_size))

#results parse
def get_throughput(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Img/sec per GPU:" not in line: continue
            throughput = eval(line.split()[-2])
            return throughput
        assert False

block_num_level = [1, 2, 4, 8]
thread_num_level = [64, 128, 256, 512]

THREAD_OPTIONS = [64, 128, 256, 512, 1024, 2048, 4096]
corresponding_block_num = [1, 1, 1, 1, 2, 4, 8]
corresponding_thread_num = [64, 128, 256, 512, 512, 512, 512]

fusion_group = [4, 4, 4, 8, 8, 4]
#100 - 32: baseline 75
#25 - 32: baseline 75
#100 - 2: baseline 14 
baseline = 100

def _get_near_integer(v):
    return int(np.round(v))

def train_bert_large(**kwargs):
    #train_cmd = 'NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_ALGO=ring FUSION_SIZE=%s FUSION_BLOCK_NUM=%s FUSION_THREAD_NUM=%s HOROVOD_TIMELINE=amoeba-tmp horovodrun --verbose --gloo --log-level INFO --network-interface "eno1" -np 2 -H localhost:1,192.168.2.3:1 python3 horovod_amoebanet.py > %s'
    #train_cmd = 'HOROVOD_FUSION_THRESHOLD=629145600 NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_ALGO=ring FUSION_SIZE=%s FUSION_BLOCK_NUM=%s FUSION_THREAD_NUM=%s HOROVOD_TIMELINE=tmp horovodrun --verbose --gloo --log-level INFO --network-interface "eno1" -np 2 -H localhost:1,192.168.2.3:1 python3 tensorflow2_synthetic_benchmark.py --batch-size 32 --model ResNet152 > %s'
    train_cmd = 'HOROVOD_FUSION_THRESHOLD=2147483648 NCCL_MIN_NCHANNELS=8 FUSION_SIZE=%s FUSION_BLOCK_NUM=%s FUSION_THREAD_NUM=%s NCCL_MAX_NCHANNELS=8 NCCL_ALGO=ring HOROVOD_TIMELINE=tmp horovodrun --verbose --gloo --log-level INFO --network-interface "lo" -np 2 -H localhost:2 python3.7 tensorflow2_synthetic_benchmark.py --model VGG16 --batch-size 32 --num-iters 32 > %s'
    fusion_size = ""
    block_num = ""
    thread_num = ""
    for gi in range(len(fusion_group)):
        if gi > 0:
            fusion_size += ","
            block_num += ","
            thread_num += ","
        gs = fusion_group[gi]
        fusion_size += str(gs)
        thread_index = _get_near_integer(kwargs["t"+str(gi)])
        block_num += str(corresponding_block_num[thread_index])
        thread_num += str(corresponding_thread_num[thread_index])
    train_cmd = train_cmd % (fusion_size, block_num, thread_num, tmp_file)
    #print(train_cmd)
    #exit()
    os.system(train_cmd)
    tp = get_throughput(tmp_file)
    print(train_cmd)
    print("result:{} imgs/sec".format(tp))
    #tp = 1
    # log the results
    log.write(train_cmd + "\n")
    log.write(str(tp) + "\n")
    log.flush()
    return (tp - baseline) * 100


# Bounded region of parameter space
pbounds = {}
for gi in range(len(fusion_group)):
    pbounds["t" + str(gi)] = (0, len(THREAD_OPTIONS) - 1e-4)


optimizer = BayesianOptimization(
    f=train_bert_large,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=init_num,
    n_iter=iter_num,
)

log.close()
