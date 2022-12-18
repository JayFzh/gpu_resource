from model import dl_model, cluster_model, dl_model_data
from interference import overlap_test as overlap
from interference import overlap_test2 as overlap2
from perf_model import overlap_perf_comm_stage as perf_model
#from scripts.run_data_fit import perf_model
from perf_model import alloc_policy
from bo import bayes_alloc
import numpy as np
import random
import time

# simple baseline policy
def random_plc(layer_num):
    thd_allocation = []
    for i in range(layer_num):
        thd_allocation.append(cluster_model.THREAD_OPTIONS[random.randint(0, len(cluster_model.THREAD_OPTIONS) - 1)])
    #thd_allocation = [128, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    return thd_allocation

# default policy
def default(model: dl_model, default_thd_option = 1024 * cluster_model.bw_incre):
    layer_num = model.layer_num
    thread_allocation = [default_thd_option for i in range(layer_num)]
    #thread_allocation = [128, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    plc = alloc_policy(thread_allocation)
    iter_time = perf_model(model, plc, debug=True)
    return thread_allocation, iter_time

# without fusion optimization
def dp(model: dl_model):
    # results
    thread_allocation = []
    opt_time = -1

    # params
    layer_num = model.layer_num

    # init dp states
    model.set_discrete_comp_time(Z)
    time_chunk = model.get_discrete_comp_time()
    states = [[[np.inf, [-1, -1], -1] for i in range(Z + 1)] for j in range(layer_num + 1)] # [opt_value, last_hop, thread_allocation]
    for j in range(Z + 1):
        states[0][j] = [sum(time_chunk[:j]), [-1, -1], -1]

    for k in range(1, layer_num + 1):
        print("Layer:{}".format(k))
        group_id = k - 1
        print(model.get_cell_comp_index(group_id))
        for n in range(Z + 1):
            comp_chunk_id = n - 1
            start_chunk_id = model.get_cell_comp_index(group_id)
            n_hat_start = start_chunk_id + 1
            opt_value = np.inf
            opt_last_hop = [-1, -1]
            opt_thread_alloc = -1
            for n_hat in range(n_hat_start, n + 1):
                n_hat_chunk_id = n_hat - 1
                for thd in cluster_model.THREAD_OPTIONS:
                    tmp_value = states[k - 1][n_hat][0]
                    comm_time, comp_time = overlap(n_hat_chunk_id + 1, comp_chunk_id, group_id, thd, model)
                    # constraint
                    '''
                    if k == layer_num and n == Z:
                        print("thd: {}, comm_time: {}".format(thd, comm_time))
                    '''
                    if comm_time >0 : comm_time += cluster_model.cluster_comm_lat()
                    if comm_time > comp_time and n != Z: continue
                    tmp_value += max([comm_time, comp_time])
                    if tmp_value + 1e-2 < opt_value:
                        #print(1)
                        opt_value = tmp_value
                        opt_last_hop = [k - 1, n_hat]
                        opt_thread_alloc = thd
            #print(opt_value)
            states[k][n] = [opt_value, opt_last_hop, opt_thread_alloc]

    opt_res = states[layer_num][Z]
    print("opt:", opt_res)
    opt_time = opt_res[0]

    for li in range(len(states)):
        print(states[li])
    # gather thread allocation
    comm_start_tp = []
    trace_state = opt_res
    last_hop = trace_state[1]
    while last_hop[0] != -1:
        thread_allocation.insert(0, trace_state[-1])
        trace_state = states[last_hop[0]][last_hop[1]]
        last_hop = trace_state[1]
        comm_start_tp.insert(0, trace_state[0])

    print("total comm (ms):", sum(model.layer_size)/ cluster_model.bandwidth * 1000.0)
    print("total comp (ms):", sum(model.layer_computation_time))
    print("comm start time:", comm_start_tp)
    print("comp end   time:", np.cumsum(model.layer_computation_time))
    assert len(comm_start_tp) == len(np.cumsum(model.layer_computation_time))
    print("dependency violation:", sum(np.array(comm_start_tp) < np.cumsum(model.layer_computation_time)))

    # validate with perf model
    print("Perf model validation---------------")
    #thread_allocation[-1] = 512
    plc = alloc_policy(thread_allocation)
    perf_model(model, plc, debug = True)
    print("------------------------------------")
    print("Random policy-----------------------")
    random_choice = random_plc(layer_num)
    print(random_choice)
    plc = alloc_policy(random_choice)
    perf_model(model, plc, debug = True)
    print("------------------------------------")
    return thread_allocation, opt_time


def dp_fusion(model: dl_model):
    assert model.layer_num >= MAX_FUSION_NUM
    # results
    thread_allocation = []
    layer_fusion = []
    opt_time = -1
    comm_start_tp = []


    # params
    layer_num = model.layer_num
    latency = cluster_model.cluster_comm_lat()
    #'''
    # init dp states
    model.set_discrete_comp_time(Z)
    time_chunk = model.get_discrete_comp_time()
    states = [[[[np.inf, [-1, -1, -1], -1, 0] for i in range(Z + 1)] for j in
              range(layer_num + 1)] for p in range(MAX_FUSION_NUM + 1)]  # [opt_value, last_hop, thread_allocation]
    for j in range(Z + 1):
        states[0][0][j] = [sum(time_chunk[:j]), [-1, -1, -1], -1, 0]

    for s in range(1, MAX_FUSION_NUM + 1):
        print("Stage:{}".format(s))
        for l in range(s, layer_num + 1): # l >= s
            layer_id = l - 1
            #print(model.get_cell_comp_index(layer_id))
            for z in range(Z + 1):
                #print("s{}-l{}-z{}".format(s, l ,z))
                comp_chunk_id = z - 1
                start_chunk_id = model.get_cell_comp_index(layer_id)
                z_hat_start = start_chunk_id + 1
                opt_value = np.inf
                opt_last_hop = [-1, -1, -1]
                opt_thread_alloc = -1
                lat_count = 0
                # start searching
                if l == layer_num: ll = l + 1 # to avoid empty group in the middle
                else: ll = l
                for l_hat in range(ll):
                    gradient_size = 0
                    for li in range(l_hat + 1, l + 1):
                        gradient_size += model.layer_size[li - 1]
                    for z_hat in range(z_hat_start, z + 1):
                        z_hat_chunk_id = z_hat - 1
                        comp_size = sum(time_chunk[z_hat_chunk_id + 1 : comp_chunk_id + 1])
                        for thd in cluster_model.THREAD_OPTIONS:
                            tmp_value = states[s - 1][l_hat][z_hat][0]
                            tmp_lat_count = states[s - 1][l_hat][z_hat][-1]
                            comm_time, comp_time = overlap2(comp_size, gradient_size, thd, model)
                            # constraint

                            if gradient_size > 0:
                                comm_time += latency  # cluster_model.latency
                                tmp_lat_count += latency
                            if comm_time > comp_time and z != Z: continue
                            tmp_value += max([comm_time, comp_time])
                            if tmp_value + 1e-2 < opt_value:
                                # print(1)
                                opt_value = tmp_value
                                opt_last_hop = [s - 1, l_hat, z_hat]
                                opt_thread_alloc = thd
                                lat_count = tmp_lat_count
                                #if s == 3: exit()
                                #print(comp_size)
                                if z == -100 and l == 5 and s == 1:
                                    print("comm size:{}  comp size:{} ".format(gradient_size, comp_size))
                                    print("Start id:{} thd:{} sum comp:{} last hop value:{}".format(z_hat_start, thd, len(time_chunk), states[s - 1][l_hat][z_hat][0]))
                                    print("Opt value:{}  s{}-l{}-z{}  last_hop:{} add value:{} comm:{} comp:{}".format(opt_value, s, l, z, opt_last_hop, max([comm_time,comp_time]),comm_time,comp_time))
                                #if s == 3 and l == 15: exit()
                # print(opt_value)
                states[s][l][z] = [opt_value, opt_last_hop, opt_thread_alloc, lat_count]

    opt_res = states[MAX_FUSION_NUM][layer_num][Z]
    print("opt:", opt_res)
    opt_time = opt_res[0]
    #exit()

    for li in range(len(states)):
        print(states[li])
    # gather thread allocation
    trace_state = opt_res
    last_hop = trace_state[1]
    tmp_li = layer_num
    while last_hop[0] != -1:
        print("xxxx:", last_hop)
        print(trace_state)
        layer_fusion.insert(0, tmp_li - last_hop[1])
        tmp_li = last_hop[1]
        thread_allocation.insert(0, trace_state[2])
        trace_state = states[last_hop[0]][last_hop[1]][last_hop[2]]
        last_hop = trace_state[1]
        comm_start_tp.insert(0, trace_state[0])


    print("layer_fusion:", layer_fusion)
    assert sum(layer_fusion) == layer_num
    print("total comm (ms):", sum(model.layer_size) / cluster_model.bandwidth * 1000.0)
    print("total comp (ms):", sum(model.layer_computation_time))
    print("comm start time:", comm_start_tp)
    model.group(layer_fusion)
    print("comp end   time:", np.cumsum(model.layer_computation_time))
    #assert len(comm_start_tp) == len(np.cumsum(model.layer_computation_time))
    #print("dependency violation:", sum(np.array(comm_start_tp) < np.cumsum(model.layer_computation_time)))

    # validate with perf model
    print("Perf model validation---------------")
    # thread_allocation[-1] = 2048
    plc = alloc_policy(thread_allocation)
    iter_time = perf_model(model, plc, debug=True)
    print("Perf iter time:{}".format(iter_time))
    print("------------------------------------")
    print("Random policy-----------------------")
    plc = alloc_policy(random_plc(len(layer_fusion)))
    perf_model(model, plc, debug=True)
    print("------------------------------------")
    return thread_allocation, opt_time, layer_fusion


# MG-WFBP algorithm
from scripts.contention_fit2 import _fitting_unoverlap_comm
def merge(tao_b, t_c, p, li):
    t_c[li] = 0
    p[li -1] = p[li] + p[li - 1]
    t_c[li - 1] =  _fitting_unoverlap_comm([p[li - 1], 1024 * cluster_model.bw_incre])

def cal_comm_start(t_c, t_b, tao_b, layer_num):
    new_tao_c = [0 for i in range(layer_num)]
    new_tao_c[layer_num - 1] = tao_b[layer_num - 1] + t_b[layer_num - 1]
    for li in range(layer_num - 2, -1 ,-1):
        new_tao_c[li] = max(new_tao_c[li + 1] + t_c[li + 1], tao_b[li] + t_b[li])
    return new_tao_c

def _disp_format(input_list):
    string = ""
    for li, k in enumerate(input_list):
        if li == len(input_list) -1: fmt = "{}"
        else: fmt = "{},"
        string += fmt.format(k)
    return string
def disp_merge_group(merged):
    size_accum = 0
    merged_size = []
    for state in merged:
        size_accum += 1
        if state == 0:
            merged_size.append(size_accum)
            size_accum = 0
    if size_accum >0: merged_size.append(size_accum)
    print("Merged group: {} -sum:{}".format(_disp_format(merged_size), sum(merged_size)))
    print(_disp_format([0 for i in range(len(merged_size))]))
    return merged_size

def mg_wfbp(model:dl_model):
    LAT = cluster_model.cluster_comm_lat() # ms

    layer_num = model.layer_num
    layer_comp_time = list(model.layer_computation_time)
    layer_comm_size = list(model.layer_size)
    layer_comm_time = []
    # get comm time
    for li in range(layer_num):
        layer_comm_time.append( _fitting_unoverlap_comm([layer_comm_size[li], 10248 * cluster_model.bw_incre ]))

    # result
    merged = [0 for i in range(layer_num)]

    t_c = layer_comm_time
    t_b = layer_comp_time
    p = layer_comm_size
    tao_b = [0 for i in range(layer_num)]
    for li in range(layer_num - 2, -1, -1):
        tao_b[li] = tao_b[li + 1] + t_b[li + 1]

    tao_c = cal_comm_start(t_c, t_b, tao_b, layer_num)
    for li in range(layer_num - 1, 0, -1):
        if (tao_b[li - 1] + t_b[li - 1])  < (LAT + tao_c[li]):
            merge(tao_b, t_c, p, li)
            tao_c = cal_comm_start(t_c, t_b, tao_b, layer_num)
            merged[li] = 1

    #print("Merged layer:{}".format(merged))
    merged_group = disp_merge_group(merged)
    return merged_group

def easgd(model:dl_model):
    LAT = cluster_model.cluster_comm_lat()  # ms
    layer_num = model.layer_num
    layer_comp_time = list(model.layer_computation_time)
    layer_comm_size = list(model.layer_size)
    layer_comm_time = []
    # get comm time
    for li in range(layer_num):
        layer_comm_time.append(_fitting_unoverlap_comm([layer_comm_size[li], 4096]))
    total_comm_time = sum(layer_comm_time) + LAT
    total_comp_time = sum(layer_comp_time)
    return total_comm_time + total_comp_time

from scripts.run_data_sample import load_model
PREPROCESS_LAYER_NUM = 40
# setting
Z = 120 # time discretize number
MAX_FUSION_NUM = 8 # max number of groups for fusion num
if __name__ == "__main__":
    bandwidth = 50 *8  # 50Gb/s is the baseline bandwidth - try getting the VGG curve first
    comp_ratio = 1.0 # comp speedup
    cluster_model.curve_bandwidth = bandwidth
    # change configs according to bandwith
    cluster_model.bw_incre = bandwidth // cluster_model.baseline_bandwidth
    cluster_model.THREAD_OPTIONS = list(np.array(cluster_model.THREAD_OPTIONS) * int(cluster_model.bw_incre))
    assert np.log2(cluster_model.bw_incre) == int(np.log2(cluster_model.bw_incre))

    test_model = [["VGG16", 32], ["VGG16", 16], ["Xception", 8], ['Bert', 2], ["ResNet152", 8]][1]
    algorithm = ["dp", "dp-fusion", "bayes", "default", "merge-wfbp", "default-random-fusion", "EASGD", "manual"][-4]
    fusion_required = ["dp", "bayes", "default", "manual"]

    # construct model
    model_name = test_model[0]
    batch_size = test_model[1]
    model_info = load_model(model_name, batch_size, prefix="./scripts/")
    pids = list(model_info.keys())
    pids.sort()
    # print(pids)
    comp_time = list(np.array([model_info[pid][1] for pid in pids]) / comp_ratio)
    # translate comm_size to cluster settings
    #comm_size = [cluster_model.cluster_comm_size(model_info[pid][0]) for pid in pids]
    comm_size = [model_info[pid][0] for pid in pids]
    model_data = dl_model_data()
    model_data.layer_size = comm_size
    model_data.layer_computation_time = comp_time
    model = dl_model(model_data=model_data, fusion_num= PREPROCESS_LAYER_NUM)

    #print(model.fusion_group)
    #exit()
    # using specified group or not
    fusion_group = None#[1, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]#[1, 1, 1, 1, 3, 20, 5]#  None#[1, 4, 27] #[9, 23]#[1, 9, 4, 2, 4, 2, 2, 2, 2, 2, 2] #[1, 6, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]#[1, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]##[4, 4, 4, 8, 8, 4]#[4 for i in range(8)] # [7, 20, 5]  [512, 512, 4096]      [2 for i in range(len(pids) // 2)]
    thread_allocation = None#[2048 for i in range(15)]#[512*4 for i in range(8)]#[1024,128,1024,128,1024,1024,512]#None
    if algorithm in fusion_required:
        assert fusion_group is not None
    else:
        assert fusion_group is None
    if fusion_group: model.group(fusion_group)

    # algorithm
    start_t = time.time()
    if algorithm == "EASGD":
        opt_time = easgd(model)
    elif algorithm == "merge-wfbp":
        merged_group = mg_wfbp(model)
        model.group(merged_group)
        thread_allocation, opt_time = default(model, default_thd_option=512*cluster_model.bw_incre)# then using default resource policy
    elif algorithm == "dp":
        thread_allocation, opt_time = dp(model)
    elif algorithm == "bayes":
        thread_allocation, opt_time = bayes_alloc(model)
    elif algorithm == "default":
        #print("xxxx",model.layer_size)
        thread_allocation, opt_time = default(model, default_thd_option=512*cluster_model.bw_incre)
    elif algorithm == "dp-fusion":
        thread_allocation, opt_time, layer_fusion = dp_fusion(model)
        print("Fusion result:{}".format(layer_fusion))
        #model.group(layer_fusion)
    elif algorithm == "default-random-fusion":
        # repead for 30 round:
        opt_res = []
        for i in range(1000):
            if i % 10 == 0: print("Iter: {}".format(i))
            model.random_fusion_model(MAX_FUSION_NUM) # random tensor fusion
            thread_allocation, opt_time = default(model, default_thd_option=1024*cluster_model.bw_incre)
            opt_res.append(opt_time)
            model = dl_model(model_data=model_data, fusion_num=PREPROCESS_LAYER_NUM)
        opt_time = np.mean(opt_res)
    elif algorithm == "manual":
        assert thread_allocation != None
        print("Perf model validation---------------")
        # thread_allocation[-1] = 2048
        plc = alloc_policy(thread_allocation)
        opt_time = perf_model(model, plc, debug=True)
        print("Perf iter time:{}".format(opt_time))
        print("------------------------------------")

    end_t = time.time()
    if thread_allocation != -1:
        print(_disp_format(thread_allocation))
        print(_disp_format([0 for i in range(len(thread_allocation))]))
        print(_disp_format([max(1, thread_allocation[i]//512) for i in range(len(thread_allocation))]))
        print(_disp_format([min(512, thread_allocation[i]) for i in range(len(thread_allocation))]))
    print(opt_time)
    print("Solution time: {}s".format(end_t - start_t))
    print("Real fusion group:{}".format(model.fusion_group))


