import os
import sys
import pickle
sys.path.append("../")
from tracing.chrome_tracing_v2 import *

# global setting
data_path = "/home/users/fzh/log/"
gpu_num = 2
# model_settings = {"VGG16": [2, 4, 8, 16, 32, 64],
#                  "ResNet152": [2, 4, 8, 16, 32, 64]}
model_settings = {"VGG16": [2, 4, 8, 16, 32, 64],
                    "MobileNetV2": [2, 4, 8, 16, 32, 64],
                    "ResNet50":[2,4,8,16,32,64]}
thread_options = [[64, 1], [128, 1], [256, 1], [512, 1],
                  [512, 2], [512, 4], [512, 8]]
fzh_thread_resnet = [
'512,512,512,512,32',
'512,512,512,512,64',
'512,512,512,512,128', 
'512,512,512,512,256',
'512,512,512,512,512',

'512,512,512,32,512',
'512,512,512,64,512',
'512,512,512,128,512',
'512,512,512,256,512',
'512,512,512,512,512',

'512,512,32,512,512',
'512,512,64,512,512',
'512,512,128,512,512',
'512,512,256,512,512',
'512,512,512,512,512',

'512,32,512,512,512',
'512,64,512,512,512',
'512,128,512,512,512',
'512,256,512,512,512',
'512,512,512,512,512',

'32,512,512,512,512',
'64,512,512,512,512',
'128,512,512,512,512',
'256,512,512,512,512',
'512,512,512,512,512']

fzh_thread_mobilenet = [
'512,512,512,32',
'512,512,512,64',
'512,512,512,128', 
'512,512,512,256',
'512,512,512,512'
]
# '512,512,32,512',
# '512,512,64,512',
# '512,512,128,512',
# '512,512,256,512',
# '512,512,512,512',

# '512,32,512,512',
# '512,64,512,512',
# '512,128,512,512',
# '512,256,512,512',
# '512,512,512,512',

# '32,512,512,512',
# '64,512,512,512',
# '128,512,512,512',
# '256,512,512,512',
# '512,512,512,512'

cmd_fmt = 'HOROVOD_FUSION_THRESHOLD=2147483648 NCCL_MIN_NCHANNELS={} NCCL_MAX_NCHANNELS={} FUSION_SIZE={} FUSION_BLOCK_NUM={} FUSION_THREAD_NUM={} HOROVOD_STREAM_ASSIGNMENT={} NCCL_ALGO=ring HOROVOD_TIMELINE={} horovodrun --verbose --gloo --log-level INFO --network-interface "lo" -np {} -H localhost:{} python3 /home/users/fzh/horovod-0.20.3/examples/tensorflow2/tensorflow2_synthetic_benchmark.py --model {} --batch-size {} --num-iters {}'
model_info_fmt = "info-{}-b{}"
max_group_num = 12
# dif files
thread_num = 32
# get tensor num
def get_tensor_num(model_name):
    # config
    min_channel = 1
    max_channel = 8
    fusion_s = 0
    fusion_b = 0
    fusion_t = 0
    log_file_path = data_path + "{}_debug".format(model_name)
    global gpu_num
    batch_size = 4
    num_iters = 5
    cmd = cmd_fmt.format(min_channel, max_channel, fusion_s, fusion_b, fusion_t,"1",log_file_path, gpu_num, gpu_num, model_name, batch_size, num_iters)
    os.system(cmd)
    # parse results
    layer_stats, layer_info = get_layer_wise_info(log_file_path)
    return len(layer_info.keys())

# def get_tensor_num(model_name,thread_assign):
#     # config
#     min_channel = 8
#     max_channel = 8
#     fusion_s    = "40,40,40,38"
#     fusion_b    = "1,1,1,1"
#     fusion_t    = thread_assign
#     stream_assign="2,3,4,5"
#     log_file_path = data_path + "{}_debug_4stream.json".format(model_name)
#     global gpu_num
#     batch_size = 16
#     num_iters = 20
#     cmd = cmd_fmt.format(min_channel, max_channel, fusion_s, fusion_b, fusion_t, stream_assign, log_file_path, gpu_num, gpu_num, model_name, batch_size, num_iters)
#     os.system(cmd)
#     # parse results
#     layer_stats, layer_info = get_layer_wise_info(log_file_path)
#     model_info_comp = fill_layer_computation(layer_stats, layer_info)
#     model_info_comm = fill_layer_communication(layer_stats,layer_info,fusion_s,thread_assign)
#     save_model(model_name, batch_size, model_info_comp)
#     # dif files
#     return len(layer_info.keys())

# profile model layers
# get no-overlap model(comm and comp)
# what i want is no comm overlap model
def get_model_info(model_name, batch_size, tensor_num):
    # config
    min_channel = 8
    max_channel = 8
    fusion_s = tensor_num
    fusion_b = 1
    fusion_t = 512
    stream_assign=1
    log_file_path = data_path + "{}_b{}_nooverlap".format(model_name, batch_size)
    global gpu_num
    num_iters = 5
    cmd = cmd_fmt.format(min_channel, max_channel, fusion_s, fusion_b, fusion_t, stream_assign, log_file_path, gpu_num, gpu_num, model_name, batch_size, num_iters)
    os.system(cmd)
    # parse results
    layer_stats, layer_info = get_layer_wise_info(log_file_path)
    model_info = fill_layer_computation(layer_stats, layer_info) # {pid: [layer size, computation time, comp_std]}
    return model_info

# generate data with default policy
def generate_default(model_name, batch_size, thread_alloc):
    # config
    block_num = thread_alloc[1]
    thread_num = thread_alloc[0]
    min_channel = block_num
    max_channel = block_num
    fusion_s = 0
    fusion_b = 0
    fusion_t = 0
    log_file_path = data_path + "{}_b{}_t{}".format(model_name, batch_size, block_num * thread_num)
    global gpu_num
    num_iters = 16
    cmd = cmd_fmt.format(min_channel, max_channel, fusion_s, fusion_b, fusion_t, log_file_path, gpu_num, gpu_num,
                         model_name, batch_size, num_iters)
    cmd = 'NCCL_NTHREADS={} '.format(thread_num) + cmd
    os.system(cmd)
    # parse results
    # layer_stats, layer_info = get_layer_wise_info(log_file_path)

def _get_fusion_groups(tensor_num):
    group_specs = []
    for group_size in range(2, max_group_num+1):
        size_per_group = tensor_num // group_size
        group_spec = [size_per_group for i in range(group_size)]
        for i in range(tensor_num - size_per_group * group_size):
            group_spec[i] += 1
        group_specs.append(group_spec)
    return group_specs

def _get_fusion_groups_unover(tensor_num):
    step_size = tensor_num // max_group_num #2
    group_specs = []
    first_group_size = 1
    while first_group_size <= tensor_num:
        group_spec = [first_group_size]
        remain_size = tensor_num - first_group_size
        if remain_size > 0: group_spec.append(remain_size)
        group_specs.append(group_spec)
        first_group_size += step_size
        if first_group_size - step_size < tensor_num and first_group_size > tensor_num:
            first_group_size = tensor_num
    return group_specs

def generate_fixed(model_name, batch_size, thread_alloc, tensor_num):
    group_specs = _get_fusion_groups(tensor_num)
    for gi, group_spec in enumerate(group_specs):
        # config
        block_num = thread_alloc[1]
        thread_num = thread_alloc[0]
        min_channel = block_num
        max_channel = block_num
        fusion_s = ""
        fusion_b = ""
        fusion_t = ""
        stream_assign = ""
        for i in range(0,len(group_spec)):
            stream_assign += str(i+2)+',' 
        stream_assign = stream_assign[:-1]
        for si, s in enumerate(group_spec):
            if si != 0:
                fusion_s += ','
                fusion_b += ','
                fusion_t += ','
            fusion_s += str(s)
            fusion_b += str(block_num)
            fusion_t += str(thread_num)
        log_file_path = data_path + "{}_b{}_t{}_g{}".format(model_name, batch_size, block_num * thread_num, gi)
        global gpu_num
        num_iters = 5
        cmd = cmd_fmt.format(min_channel, max_channel, fusion_s, fusion_b, fusion_t, stream_assign,log_file_path, gpu_num, gpu_num,
                             model_name, batch_size, num_iters)
        print(cmd)
        os.system(cmd)
        


# save the model data
def save_model(model_name, batch_size, info):
    file_path = data_path + model_info_fmt.format(model_name, batch_size)
    with open(file_path, 'wb') as f:
        pickle.dump(info, f)


if __name__ == "__main__":
    for name in model_settings.keys():
        tensor_num = get_tensor_num(name)
        for batch_size in model_settings[name]:
            model_info = get_model_info(name, batch_size, tensor_num)
            assert len(model_info) == tensor_num
            save_model(name, batch_size, model_info)
            for thread_alloc in thread_options:
                generate_fixed(name, batch_size, thread_alloc, tensor_num)
  
            
                
        # for batch_size in model_settings[name]:
        #     model_info = get_model_info(name, batch_size, tensor_num)
        #     assert len(model_info) == tensor_num
        #     save_model(name, batch_size, model_info)
            # for thread_alloc in thread_options:
            #     #generate_default(name, batch_size, thread_alloc)
            #     generate_fixed(name, batch_size, thread_alloc, tensor_num)