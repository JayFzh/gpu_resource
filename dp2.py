from model import dl_model, cluster_model, dl_model_data, THREAD_OPTIONS
from interference import overlap_test as overlap
from interference import overlap_test2 as overlap2
from perf_model import overlap_perf_comm_stage as perf_model
from perf_model import alloc_policy
from bo import bayes_alloc
import numpy as np
import random
import time


# setting
Z = 100 # time discretize number
MAX_FUSION_NUM = 20 # max number of groups for fusion num

# simple baseline policy
def random_plc(layer_num):
    thd_allocation = []
    for i in range(layer_num):
        thd_allocation.append(THREAD_OPTIONS[random.randint(0, len(THREAD_OPTIONS) - 1)])
    #thd_allocation = [128, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    return thd_allocation


# default policy
def default(model: dl_model, default_thd_option = 1024):
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
                for thd in THREAD_OPTIONS:
                    tmp_value = states[k - 1][n_hat][0]
                    comm_time, comp_time = overlap(n_hat_chunk_id + 1, comp_chunk_id, group_id, thd, model)
                    # constraint
                    '''
                    if k == layer_num and n == Z:
                        print("thd: {}, comm_time: {}".format(thd, comm_time))
                    '''
                    if comm_time > comp_time and n != Z: continue
                    tmp_value += max([comm_time, comp_time])
                    if tmp_value < opt_value:
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

    # params
    layer_num = model.layer_num

    # init dp states
    model.set_discrete_comp_time(Z)
    time_chunk = model.get_discrete_comp_time()
    states = [[[[np.inf, [-1, -1, -1], -1] for i in range(Z + 1)] for j in
              range(layer_num + 1)] for p in range(MAX_FUSION_NUM + 1)]  # [opt_value, last_hop, thread_allocation]
    for j in range(Z + 1):
        states[0][0][j] = [sum(time_chunk[:j]), [-1, -1, -1], -1]

    for s in range(1, MAX_FUSION_NUM + 1):
        print("Stage:{}".format(s))
        for l in range(s, layer_num + 1): # l >= s
            layer_id = l - 1
            #print(model.get_cell_comp_index(layer_id))
            for z in range(Z + 1):
                comp_chunk_id = z - 1
                start_chunk_id = model.get_cell_comp_index(layer_id)
                z_hat_start = start_chunk_id + 1
                opt_value = np.inf
                opt_last_hop = [-1, -1, -1]
                opt_thread_alloc = -1
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
                        for thd in THREAD_OPTIONS:
                            tmp_value = states[s - 1][l_hat][z_hat][0]
                            comm_time, comp_time = overlap2(comp_size, gradient_size, thd, model)
                            # constraint
                            '''
                            if k == layer_num and n == Z:
                                print("thd: {}, comm_time: {}".format(thd, comm_time))
                            '''
                            if comm_time > comp_time and z != Z: continue
                            tmp_value += max([comm_time, comp_time])
                            if gradient_size > 0: tmp_value += cluster_model.latency
                            if tmp_value < opt_value:
                                # print(1)
                                opt_value = tmp_value
                                opt_last_hop = [s - 1, l_hat, z_hat]
                                opt_thread_alloc = thd
                                #if s == 3: exit()
                                #print(comp_size)
                                #print("Start id:{} thd:{} sum comp:{} ".format(z_hat_start, thd, len(time_chunk)))
                                #print("Opt value:{}  s{}-l{}-z{}  last_hop:{} add value:{} comm:{} comp:{}".format(opt_value, s, l, z, opt_last_hop, max([comm_time,comp_time]),comm_time,comp_time))
                                #if s == 3 and l == 15: exit()
                # print(opt_value)
                states[s][l][z] = [opt_value, opt_last_hop, opt_thread_alloc]

    opt_res = states[MAX_FUSION_NUM][layer_num][Z]
    print("opt:", opt_res)
    opt_time = opt_res[0]
    #exit()

    for li in range(len(states)):
        print(states[li])
    # gather thread allocation
    comm_start_tp = []
    trace_state = opt_res
    last_hop = trace_state[1]
    tmp_li = layer_num
    while last_hop[0] != -1:
        print("xxxx:", last_hop)
        print(trace_state)
        layer_fusion.insert(0, tmp_li - last_hop[1])
        tmp_li = last_hop[1]
        thread_allocation.insert(0, trace_state[-1])
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
    assert len(comm_start_tp) == len(np.cumsum(model.layer_computation_time))
    print("dependency violation:", sum(np.array(comm_start_tp) < np.cumsum(model.layer_computation_time)))

    # validate with perf model
    print("Perf model validation---------------")
    # thread_allocation[-1] = 2048
    plc = alloc_policy(thread_allocation)
    perf_model(model, plc, debug=True)
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
    t_c[li - 1] =  _fitting_unoverlap_comm([p[li - 1], 1024])

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

def mg_wfbp(model:dl_model):
    LAT = 0.01 # ms

    layer_num = model.layer_num
    layer_comp_time = model.layer_computation_time
    layer_comm_size = model.layer_size
    layer_comm_time = []
    # get comm time
    for li in range(layer_num):
        layer_comm_time.append( _fitting_unoverlap_comm([layer_comm_size[li], 1024]))

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
    disp_merge_group(merged)
    return merged


from scripts.run_data_sample import load_model
PREPROCESS_LAYER_NUM = 32
if __name__ == "__main__":
    test_model = [["VGG16", 32], ["VGG16", 4],["ResNet152", 32]][1]
    algorithm = ["dp", "dp-fusion", "bayes", "default", "random", "merge-wfbp"][-1]

    # construct model
    model_name = test_model[0]
    batch_size = test_model[1]
    model_info = load_model(model_name, batch_size, prefix="./scripts/")
    pids = list(model_info.keys())
    pids.sort()
    # print(pids)
    comp_time = [model_info[pid][1] for pid in pids]
    comm_size = [model_info[pid][0] for pid in pids]
    model_data = dl_model_data()
    model_data.layer_size = comm_size
    model_data.layer_computation_time = comp_time
    model = dl_model(model_data=model_data, fusion_num= PREPROCESS_LAYER_NUM)

    fusion_group = None#[1,6,1,1,1,2,2,1,1,2,1,1,1,1,1,1,2,2,2,2]#None #[4, 4, 4, 8, 8, 4]#[4 for i in range(8)] # [7, 20, 5]  [512, 512, 4096]      [2 for i in range(len(pids) // 2)]
    if fusion_group: model.group(fusion_group)

    # algorithm
    start_t = time.time()
    if algorithm == "merge-wfbp":
        merged_group = mg_wfbp(model)
        exit()

    if algorithm == "dp":
        thread_allocation, opt_time = dp(model)
    elif algorithm == "bayes":
        thread_allocation, opt_time = bayes_alloc(model)
    elif algorithm == "default":
        thread_allocation, opt_time = default(model)
    elif algorithm == "dp-fusion":
        thread_allocation, opt_time, layer_fusion = dp_fusion(model)
        print("Fusion result:{}".format(layer_fusion))
        model.group(layer_fusion)
    end_t = time.time()
    print(_disp_format(thread_allocation))
    print(_disp_format([0 for i in range(len(thread_allocation))]))
    print(opt_time)
    print("Solution time: {}s".format(end_t - start_t))
    print("Real fusion group:{}".format(model.fusion_group))



# test 0
if __name__ == "__main__x":
    test_mode = ["random", "layer10", "layer10-2", "layer20", "layer30", "layer30-2", "layer40", "layer40-2", "layer100-fix-fusion", "layer100"][-2]
    algorithm = ["dp", "dp-fusion", "bayes", "default", "random"][0]
    # random plc
    if test_mode == "random":
        model = dl_model(random_layer_num = 100, random_size_range=(1, 1000), random_comp_range=(1,100))
    elif test_mode == "layer10":
        model_data = dl_model_data()
        model_data.layer_size = [34.966, 124.876, 257.743, 131.869, 888.112, 174.826, 24.976, 154.846, 606.394, 725.275]
        model_data.layer_computation_time = [80.69500000000001, 77.131, 5.455, 72.181, 4.366, 76.834, 46.738, 6.445, 51.985, 8.029]
        model = dl_model(model_data = model_data)
    elif test_mode == "layer10-2":
        model_data = dl_model_data()
        model_data.layer_size = [382.618, 2.998, 101.899, 738.262, 660.34, 333.66700000000003, 216.784, 99.90100000000001, 888.112, 939.0609999999999]
        model_data.layer_computation_time = [72.181, 13.078, 94.55499999999999, 57.42999999999999, 73.567, 46.045, 1.891, 99.901, 34.660000000000004, 43.372]
        model = dl_model(model_data = model_data)
    elif test_mode == "layer20":
        model_data = dl_model_data()
        model_data.layer_size = [153.847, 770.23, 198.80200000000002, 994.006, 532.4680000000001, 73.92699999999999, 839.161, 734.266, 938.0619999999999, 428.572, 834.1659999999999, 983.0169999999999, 600.4, 68.932, 737.263, 35.965, 613.387, 998.002, 78.922, 91.90899999999999]
        model_data.layer_computation_time = [93.56500000000001, 48.916, 19.216, 38.719, 8.821, 77.428, 49.51, 95.05, 82.675, 83.071, 46.243, 62.875, 53.668000000000006, 6.346, 6.9399999999999995, 53.569, 85.051, 65.548, 80.79400000000001, 56.638000000000005]
        model = dl_model(model_data = model_data)
    elif test_mode == "layer30":
        model_data = dl_model_data()
        model_data.layer_size = [424.57599999999996, 166.834, 910.09, 497.503, 77.923, 672.3280000000001, 277.723, 931.0690000000001, 853.1469999999999, 519.481, 817.183, 573.4269999999999, 661.339, 580.42, 402.598, 527.4730000000001, 12.988, 831.169, 629.371, 388.612, 133.86700000000002, 709.2909999999999, 707.293, 98.902, 314.686, 575.425, 496.504, 814.1859999999999, 773.227, 839.161]
        model_data.layer_computation_time = [75.349, 62.083, 14.167000000000002, 6.148, 33.868, 90.10000000000001, 81.982, 95.941, 32.878, 3.574, 48.025, 97.822, 58.221999999999994, 95.941, 84.06099999999999, 88.714, 51.391, 46.837, 78.319, 39.412, 38.62, 38.224, 9.217, 36.739, 75.745, 73.072, 92.971, 11.791, 82.87299999999999, 15.453999999999999]
        model = dl_model(model_data = model_data)
    elif test_mode == "layer30-2":
        model_data = dl_model_data()
        model_data.layer_size = [675.325, 734.266, 660.34, 727.273, 22.977999999999998, 884.116, 12.988, 888.112, 743.257, 151.849, 399.601, 855.145, 248.752, 379.621, 830.17, 621.379, 518.482, 304.69599999999997, 664.336, 494.506, 966.034, 894.106, 177.82299999999998, 960.04, 874.126, 789.211, 440.56, 192.808, 82.918, 232.768]
        model_data.layer_computation_time =[40.996, 76.339, 68.914, 55.94500000000001, 8.029, 10.405, 30.205, 54.757000000000005, 17.335, 33.571, 36.144999999999996, 97.92099999999999, 91.783, 14.167000000000002, 48.421, 33.175000000000004, 59.608, 68.81500000000001, 14.761000000000001, 5.257, 58.321, 11.494, 29.709999999999997, 48.421, 43.471, 55.54900000000001, 1.099, 31.591, 11.197, 74.656]
        model = dl_model(model_data = model_data)
    elif test_mode == "layer40":
        model_data = dl_model_data()
        model_data.layer_size = [229.77100000000002, 74.926, 75.925, 190.81, 743.257, 145.855, 571.429, 80.92, 983.0169999999999, 948.0519999999999, 313.687, 550.45, 70.93, 859.141, 587.413, 144.856, 575.425, 638.362, 739.261, 359.64099999999996, 421.579, 466.53400000000005, 396.60400000000004, 671.3290000000001, 840.16, 976.024, 951.049, 331.66900000000004, 229.77100000000002, 95.905, 546.4540000000001, 548.452, 274.726, 220.78, 730.27, 284.71599999999995, 943.0569999999999, 915.085, 841.159, 208.792]
        model_data.layer_computation_time = [15.949, 65.647, 52.579, 75.547, 9.019, 66.736, 61.092999999999996, 11.989, 17.632, 3.376, 21.394, 75.052, 27.829, 5.257, 89.011, 37.036, 30.898, 78.81400000000001, 91.58500000000001, 34.858000000000004, 6.247, 22.582, 98.317, 13.474, 58.519, 18.622, 48.123999999999995, 1.5939999999999999, 27.532, 50.302, 76.438, 84.259, 76.24, 59.509, 20.404, 33.373000000000005, 31.096, 6.445, 44.065, 54.163000000000004]
        model = dl_model(model_data = model_data)
    elif test_mode == "layer40-2":
        model_data = dl_model_data()
        model_data.layer_size = [139.86100000000002, 262.738, 540.46, 754.246, 763.237, 708.2919999999999, 302.698, 196.804, 698.3019999999999, 251.749, 684.316, 442.558, 449.551, 864.136, 132.868, 913.087, 182.81799999999998, 621.379, 358.642, 947.053, 827.173, 446.55400000000003, 215.785, 460.54, 889.111, 807.1930000000001, 283.717, 961.039, 939.0609999999999, 570.43, 695.305, 51.949, 793.207, 839.161, 13.987, 425.575, 52.948, 631.369, 316.684, 435.565]
        model_data.layer_computation_time = [14.563, 12.187000000000001, 76.735, 84.556, 98.218, 66.538, 33.076, 97.03, 64.55799999999999, 90.892, 70.201, 67.033, 79.705, 70.597, 56.53900000000001, 66.637, 66.142, 44.659, 98.119, 74.557, 83.368, 46.936, 82.17999999999999, 89.209, 65.944, 20.503, 98.416, 94.85199999999999, 63.073, 34.165, 97.822, 83.566, 77.626, 26.938000000000002, 12.484, 14.365, 57.23199999999999, 51.094, 43.174, 35.848]
        model = dl_model(model_data = model_data)
    elif "layer100" in test_mode:
        model_data = dl_model_data()
        model_data.layer_size = [421.579, 678.322, 20.98, 859.141, 454.546, 109.891, 51.949, 823.1769999999999, 258.742, 457.543, 546.4540000000001, 56.944,
                                 733.2669999999999, 273.72700000000003, 850.15, 327.673, 347.65299999999996, 467.533, 310.69, 667.3330000000001, 555.445, 712.288,
                                 376.624, 488.512, 983.0169999999999, 653.347, 774.226, 215.785, 1.999, 444.556, 949.0509999999999, 124.876, 214.786, 145.855, 90.91,
                                 95.905, 790.21, 800.2, 529.471, 577.423, 512.488, 709.2909999999999, 208.792, 903.097, 445.555, 264.736, 935.065, 611.389, 283.717,
                                 459.541, 566.434, 520.48, 338.66200000000003, 979.021, 842.158, 246.754, 465.535, 660.34, 534.466, 530.47, 559.441, 881.119, 490.51,
                                 84.91600000000001, 180.82, 525.475, 463.53700000000003, 765.235, 488.512, 52.948, 576.424, 930.07, 958.0419999999999, 120.88, 425.575,
                                 67.933, 268.732, 537.4630000000001, 206.79399999999998, 784.216, 188.812, 735.265, 232.768, 737.263, 736.264, 692.308, 999.001, 15.985,
                                 373.627, 773.227, 599.401, 438.562, 443.557, 28.972, 645.355, 465.535, 91.90899999999999, 723.2769999999999, 400.6, 836.164][:40]
        model_data.layer_computation_time = [91.288, 12.385, 58.717, 27.73, 19.216, 44.263, 98.812, 87.13, 48.717999999999996, 2.089, 87.625, 73.27, 80.299, 28.917999999999996,
                                             70.696, 53.866, 7.534000000000001, 55.74700000000001, 22.186, 96.634, 32.878, 44.56, 36.442, 26.443, 88.021, 89.902, 28.225,
                                             57.42999999999999, 18.523, 87.526, 38.125, 74.359, 55.846000000000004, 11.296, 49.708, 38.818, 55.153000000000006, 58.221999999999994,
                                             91.486, 43.272999999999996, 79.60600000000001, 32.581, 68.81500000000001, 26.443, 84.16, 46.639, 61.092999999999996, 99.208, 38.224,
                                             30.601, 66.043, 29.214999999999996, 1.792, 79.804, 32.977000000000004, 77.72500000000001, 36.937, 59.013999999999996, 51.292,
                                             27.235000000000003, 22.78, 26.839000000000002, 95.64399999999999, 35.452, 28.225, 10.207, 26.938000000000002, 63.964, 13.474,
                                             80.39800000000001, 52.975, 79.309, 73.963, 96.337, 77.824, 18.82, 3.178, 27.235000000000003, 73.864, 97.624, 19.81, 53.569,
                                             88.318, 24.958, 34.957, 43.272999999999996, 81.19000000000001, 39.115, 99.208, 73.765, 29.511999999999997, 60.300999999999995,
                                             2.485, 19.711, 31.69, 71.28999999999999, 4.7620000000000005, 69.70599999999999, 37.036, 42.382][:40]

        if test_mode == "layer100-fix-fusion":
            model = dl_model(model_data = model_data, fusion_num = MAX_FUSION_NUM)
            #group = [5, 5, 4, 5, 7, 8, 4, 2]#[5, 5, 3, 4, 4, 4, 7, 8]#[4, 12, 8, 8, 8]
            #model = dl_model(model_data = model_data)
            #model.group(group)
        elif test_mode == "layer100":
            model = dl_model(model_data = model_data)

    # algorithm
    start_t = time.time()
    if algorithm == "dp":
        thread_allocation, opt_time = dp(model)
    elif algorithm == "bayes":
        thread_allocation, opt_time = bayes_alloc(model)
    elif algorithm == "default":
        thread_allocation, opt_time = default(model)
    elif algorithm == "dp-fusion":
        thread_allocation, opt_time, layer_fusion = dp_fusion(model)
        print(layer_fusion)
    end_t = time.time()
    print(thread_allocation)
    print(opt_time)
    print("Solution time: {}s".format(end_t - start_t) )


# improvement analysis
if __name__ == "__main__x":
    test_mode = ["random", "layer10",  "layer20", "layer10-2", "layer20-2"][4]
    algorithm = ["dp", "dp-fusion", "bayes", "default", "random"][1]
    # random plc
    if test_mode == "random":
        model = dl_model(random_layer_num=10, random_size_range=(1, 1500), random_comp_range=(1, 100))
    elif test_mode == "layer10":
        model_data = dl_model_data()
        model_data.layer_size = [903.097, 958.0419999999999, 552.4480000000001, 835.165, 243.757, 749.251, 650.35, 482.518, 586.414, 447.553]
        model_data.layer_computation_time = [31.294, 96.139, 79.507, 68.122, 95.941, 22.087, 84.75399999999999, 75.052, 88.318, 98.614]
        model = dl_model(model_data=model_data)
    elif test_mode == "layer20":
        model_data = dl_model_data()
        model_data.layer_size = [425.575, 268.732, 544.456, 816.184, 487.513, 533.467, 937.0630000000001, 94.906, 326.67400000000004, 931.0690000000001, 529.471, 987.013, 610.39, 998.002, 168.83200000000002, 435.565, 536.464, 194.806, 922.0780000000001, 347.65299999999996]
        model_data.layer_computation_time = [34.561, 79.012, 51.589, 15.751, 87.031, 69.904, 49.213, 2.287, 50.896, 71.785, 62.578, 98.713, 53.371, 60.994, 46.837, 83.368, 91.387, 67.33, 73.17099999999999, 39.61]
        model = dl_model(model_data=model_data)
    elif test_mode == "layer10-2":
        model_data = dl_model_data()
        model_data.layer_size = [933.378, 266.323, 521.1529999999999, 1240.673, 107.42899999999999, 1098.268, 477.682, 973.851, 287.309, 1246.6689999999999]
        model_data.layer_computation_time = [30.997, 35.550999999999995, 35.056, 18.919, 75.448, 94.753, 46.342, 48.619, 55.94500000000001, 58.717]
        model = dl_model(model_data=model_data)
    elif test_mode == "layer20-2":
        model_data = dl_model_data()
        model_data.layer_size = [522.6519999999999, 638.0749999999999, 72.952, 53.465, 1002.3320000000001, 503.165, 1218.188, 1291.639, 233.345, 419.22100000000006, 177.88199999999998, 1173.218, 27.982, 856.929, 411.72600000000006, 1185.21, 284.311, 372.752, 1197.202, 471.686]
        model_data.layer_computation_time = [32.581, 57.23199999999999, 62.083, 81.982, 46.144, 68.32000000000001, 27.334000000000003, 74.161, 61.291, 19.711, 58.717, 41.491, 2.584, 92.278, 64.855, 46.738, 93.367, 73.963, 28.225, 67.132]
        model = dl_model(model_data=model_data)
        #model = dl_model(model_data=model_data, fusion_num=MAX_FUSION_NUM)

    # algorithm
    start_t = time.time()
    if algorithm == "dp":
        thread_allocation, opt_time = dp(model)
    elif algorithm == "bayes":
        thread_allocation, opt_time = bayes_alloc(model)
    elif algorithm == "default":
        thread_allocation, opt_time = default(model, default_thd_option = 1024)
    elif algorithm == "dp-fusion":
        thread_allocation, opt_time, layer_fusion = dp_fusion(model)
        print(layer_fusion)
    end_t = time.time()
    print(thread_allocation)
    print(opt_time)
    print("Solution time: {}s".format(end_t - start_t))