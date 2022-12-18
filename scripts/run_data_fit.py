import numpy as np
from scripts.contention_fit2 import func_dnn as predict_func
from scripts.contention_fit2 import func_dnn_group as predict_func_group
from scripts.contention_fit2 import func_fitting as predict_func
from scripts.contention_fit2 import func_fitting_group as predict_func_group
from scripts.run_data_sample import load_model
from tracing.chrome_tracing import *

raw_data = "./fitting_data/unoverlap-comm"

def preprocess():
    save_file = raw_data + "-processed"
    res= {}
    bandwidth = 0
    count = 0
    with open(raw_data, 'r') as f:
        for line in f.readlines():
            if len(line) < 2: continue
            entries = line.split()
            comm_time = eval(entries[0])
            comp_time = eval(entries[1])
            grad_size = np.round(eval(entries[2]), 2)
            comp_size = np.round(eval(entries[3]), 2)
            thread_num = eval(entries[4])
            bandwidth = eval(entries[5])
            if thread_num not in res.keys():
                res[thread_num] = {}
            if grad_size not in res[thread_num].keys():
                res[thread_num][grad_size] = {}
            if comp_size not in res[thread_num][grad_size].keys():
                res[thread_num][grad_size][comp_size] = [[], []]
            res[thread_num][grad_size][comp_size][0].append(comm_time)
            res[thread_num][grad_size][comp_size][1].append(comp_time)
            count += 1
            #print(len(res[thread_num][grad_size][comp_size][0]))

    count2 = 0
    with open(save_file, 'w') as f:
        for thread_num in res.keys():
            for grad_size in res[thread_num].keys():
                for comp_size in res[thread_num][grad_size].keys():
                    count2 += 1
                    sample_num = len(res[thread_num][grad_size][comp_size][0])
                    comm_time = np.mean(res[thread_num][grad_size][comp_size][0])
                    comp_time = np.mean(res[thread_num][grad_size][comp_size][1])
                    entry_fmt = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
                    entry = entry_fmt.format(comm_time, comp_time, grad_size, comp_size, thread_num, bandwidth,sample_num)
                    print(entry)
                    if comp_size <= comp_time and sample_num >= 5 and (comp_time / comp_size < 50 or comp_size == 0):
                        f.write(entry)
    print("{} in total.".format(count))
    print(count2)


def preprocess_add_nooverlap():
    save_file = raw_data + "-nooverlap"
    res = {}
    bandwidth = 0
    count = 0
    with open(raw_data, 'r') as f:
        for line in f.readlines():
            if len(line) < 2: continue
            entries = line.split()
            comm_time = eval(entries[0])
            comp_time = eval(entries[1])
            grad_size = np.round(eval(entries[2]), 2)
            comp_size = np.round(eval(entries[3]), 2)
            thread_num = eval(entries[4])
            bandwidth = eval(entries[5])
            if thread_num not in res.keys():
                res[thread_num] = {}
            if grad_size not in res[thread_num].keys():
                res[thread_num][grad_size] = {}
            if comp_size not in res[thread_num][grad_size].keys():
                res[thread_num][grad_size][comp_size] = [[], []]
            res[thread_num][grad_size][comp_size][0].append(comm_time)
            res[thread_num][grad_size][comp_size][1].append(comp_time)
            count += 1
            # print(len(res[thread_num][grad_size][comp_size][0]))

    count2 = 0
    with open(save_file, 'w') as f:
        for thread_num in res.keys():
            for grad_size in res[thread_num].keys():
                for comp_size in res[thread_num][grad_size].keys():
                    count2 += 1
                    sample_num = len(res[thread_num][grad_size][comp_size][0])
                    comm_time = np.mean(res[thread_num][grad_size][comp_size][0])
                    comp_time = np.mean(res[thread_num][grad_size][comp_size][1])
                    entry_fmt = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
                    entry = entry_fmt.format(comm_time, comp_time, grad_size, comp_size, thread_num, bandwidth,
                                             sample_num)
                    if comp_size <= comp_time and sample_num >= 5 and comp_time / comp_size < 50:
                        f.write(entry)
                        # add nooverlap computation entry
                        no_overlap_entry = entry_fmt.format(0, comp_size, 0, comp_size, thread_num, bandwidth,
                                                 sample_num)
                        f.write(no_overlap_entry)
    print("{} in total.".format(count))
    print(count2)

# perf model validation
def perf_plot(sim_result, iter_result):
    pass

def group_tensor(model_info, group_spec):
    fusion_info = {}
    pids = list(model_info.keys())
    pids.sort()
    p_count = 0
    fusion_id = 0
    fusion_info_entry = np.array([0 for i in range(len(model_info[pids[0]]))])
    total_comp_time = 0
    for i, pid in enumerate(pids):
        total_comp_time += model_info[pid][1]
        p_count += 1
        fusion_info_entry = fusion_info_entry + model_info[pid]
        #print("{} - {}".format(pid, fusion_id))
        if p_count == group_spec[fusion_id]:
            fusion_info[fusion_id] = list(fusion_info_entry)
            fusion_id += 1
            p_count = 0
            fusion_info_entry = np.array([0 for i in range(len(model_info[pids[0]]))])
    print("Total comptime:{}ms".format(total_comp_time))
    print("fused info:{}".format(fusion_info))
    return fusion_info

def perf_model(model_info, thread_allocation, debug = True):
    pids = list(model_info.keys())
    pids.sort()
    comp_period = [0 for i in range(len(pids))]
    comm_period = [0 for i in range(len(pids))]

    assert len(thread_allocation) == len(pids)
    layer_num = len(pids)
    stage_num = layer_num + 1
    stage_time = [0 for i in range(stage_num)]
    comp_time = [model_info[pid][1] for pid in pids]
    comm_sizes = [model_info[pid][0] for pid in pids]

    comp_index = 0
    comp_remain = 0
    comm_remain = 0
    #statistics
    total_overlap_comp_time = 0

    # compute the stage time layer by layer
    lat_to_add = 0
    for si in range(stage_num):
        li = si - 1
        if si == 0:
            stage_time[si] = comp_time[comp_index]
            comp_index += 1
            # add time point
            comp_period[0] = [0, stage_time[0]]
            continue
        print("New communication")
        # for stats
        #comm_waited = False
        comm_waited_time = 0

        last_stage_end = stage_time[si - 1]
        current_stage_end = last_stage_end
        comm_remain = comm_sizes[li]
        lat_to_add = 1
        # account for latency

        while comm_remain > 0:
            if comp_remain == 0 and comp_index < layer_num:
                comp_remain = comp_time[comp_index]
                comp_index += 1

            overlap_comm = predict_func(comm_remain, comp_remain, thread_allocation[li], type = "comm")
            overlap_comp = predict_func(comm_remain, comp_remain, thread_allocation[li], type = "comp")

            if lat_to_add:
                lat_to_add = 0
                overlap_comm += cluster_model.cluster_comm_lat()

            if overlap_comm <= overlap_comp:
                comm_remain = 0
                # confirm stage time
                assert  comp_index >= (li + 2) or comp_index == layer_num
                if comp_index == (li + 2):
                    # wait for the comp to end
                    current_stage_end += overlap_comp
                    # stats
                    total_overlap_comp_time += overlap_comp
                    comp_remain = 0
                    # add time point
                    comp_period[comp_index - 1] = [comp_period[comp_index - 2][1], current_stage_end]
                    comm_waited_time = overlap_comp - overlap_comm
                else:
                    # get remain comp time
                    current_stage_end += overlap_comm
                    # stats
                    total_overlap_comp_time += overlap_comm
                    unoverlapped = overlap_comp - overlap_comm
                    comp_remain = unoverlapped
            else:
                if comp_index == layer_num:
                    current_stage_end += overlap_comm
                    if comp_remain > 0:
                        total_overlap_comp_time += overlap_comp
                        # add time point
                        comp_period[comp_index - 1] = [comp_period[comp_index - 2][1], current_stage_end - overlap_comm + overlap_comp]
                    comp_remain = 0
                    break
                # get next comp
                current_stage_end += overlap_comp
                # stats
                total_overlap_comp_time += overlap_comp
                unoverlap_comm = predict_func(comm_remain, 0, thread_allocation[li], type = "comm")
                comm_remain *= (overlap_comm - overlap_comp) / unoverlap_comm
                comp_remain = 0
                # add time point
                comp_period[comp_index - 1] = [comp_period[comp_index - 2][1], current_stage_end]
        stage_time[si] = current_stage_end

        comm_period[li] = [last_stage_end, current_stage_end - comm_waited_time]

    # debug
    if debug: print(stage_time)
    print("[stats] total computation after overlap:{}".format(total_overlap_comp_time))
    #return stage_time[-1]
    return comm_period, comp_period, stage_time[-1]


# todo
def perf_model_v2(model_info, thread_allocation):
    comp_period = []
    comm_period = []
    pids = list(model_info.keys())
    pids.sort()
    t1 = [0 for i in range(len(pids) + 1)]
    t2 = [0 for i in range(len(pids) + 1)]
    t3 = [0 for i in range(len(pids) + 1)]
    s1 = [0 for i in range(len(pids) + 1)]
    s2 = [0 for i in range(len(pids) + 1)]
    f2 = [0 for i in range(len(pids) + 1)]
    f3 = [0 for i in range(len(pids) + 1)]
    for i in range(len(pids)):
        f = model_info[pids[i]]
        thread_num = 0
        if i > 0: thread_num = thread_allocation[i - 1]
        comm_time = predict_func(s1[i], f, thread_allocation[i - 1])
        comp_time = predict_func()
        t1[i] = min(comm_time, comp_time)

# process measurment results
def avg_measurement(iter_infos, layer_info, fusion_group):
    iter_ids = list(iter_infos.keys())
    iter_ids.sort()
    comm_length = [[] for i in range(len(fusion_group))]
    comp_length = [[] for i in range(len(fusion_group))]
    for id in iter_ids:
        iter_info = iter_infos[id]
        fusion_iter, _ = tensor_fusion(iter_info, layer_info, fusion_group)
        fusion_ids = list(fusion_iter.keys())
        fusion_ids.sort()
        for i, fid in enumerate(fusion_ids):
            comm_time = fusion_iter[fid]["NCCL_ALLREDUCE"][1] - fusion_iter[fid]["NCCL_ALLREDUCE"][0]
            if i == 0: comp_time = fusion_iter[fid]["WAIT_FOR_OTHER_TENSOR_DATA"][1] - fusion_iter[fid]["WAIT_FOR_OTHER_TENSOR_DATA"][0]
            else: comp_time = fusion_iter[fid]["WAIT_FOR_OTHER_TENSOR_DATA"][1] - fusion_iter[fid]["WAIT_FOR_DATA"][0]
            comm_length[i].append(comm_time)
            comp_length[i].append(comp_time)

    avg_comm = []
    avg_comp = []
    comm_std = []
    comp_std = []
    #disp results
    for i in range(len(fusion_group)):
        avg_comm.append(np.mean(comm_length[i]))
        avg_comp.append(np.mean(comp_length[i]))
        comm_std.append(np.std(comm_length[i]))
        comp_std.append(np.std(comp_length[i]))
    print("avg comm:", avg_comm)
    print("comm std:", comm_std)
    print("avg comp:", avg_comp)
    print("comp std:", comp_std)

    comm_periods = []
    comp_periods = []
    start_t = 0
    for i in range(len(fusion_group)):
        comp_period = [start_t, start_t + avg_comp[i]]
        start_t += avg_comp[i]
        comp_periods.append(comp_period)
    comm_t = 0
    for i in range(len(fusion_group)):
        comm_period = [max(comm_t, comp_periods[i][1]), max(comm_t, comp_periods[i][1]) + avg_comm[i]]
        comm_t = max(comm_t, comp_periods[i][1]) + avg_comm[i]
        comm_periods.append(comm_period)
    return comm_periods, comp_periods


def compare_by_plot(sim_comm, sim_comp, comm, comp):
    fusion_num = len(sim_comm)
    y = [i for i in range(fusion_num)]
    y.reverse()
    colors = ["red", "blue"]
    marks = ["o", "*"]
    lstyle = ["-", "--"]
    data = [
        [sim_comp, sim_comm],
        [comp, comm]
    ]
    labels = [
        ["sim_comp", "sim_comm"],
        ["comp", "comm"]
    ]
    print("method1: {}s".format(data[0][-1][-1][-1]))
    print("method2: {}s".format(data[1][-1][-1][-1]))

    plt.figure()
    for did, entry in enumerate(data):
        for op_id, op in enumerate(entry):
            for fid, period in enumerate(op):
                #ys = [y[fid] for i in range(len(period))]
                ys = [y[fid] - 0.2 * did for i in range(len(period))]
                if fid == 0: plt.plot(period, ys, color = colors[did], marker = marks[did], linestyle = lstyle[op_id], label = labels[did][op_id])
                else: plt.plot(period, ys, color = colors[did], marker = marks[did], linestyle = lstyle[op_id])
    plt.legend()
    plt.tight_layout()
    plt.show()

def perf_validate(model_name, batch_size, fusion_group, thread_allocation, measurement_file_mark = 'test256'):
    model_info = load_model(model_name, batch_size)
    grouped_model_info = group_tensor(model_info, fusion_group)
    sim_comm, sim_comp, total_time = perf_model(grouped_model_info, thread_allocation)
    # get real measurements
    if measurement_file_mark is not None:
        if measurement_file_mark != "fake":
            result_file = "./fitting_data/perf_{}".format(measurement_file_mark)
            iter_infos, layer_info = get_layer_wise_info(result_file)
            #fusion_iters, _ = tensor_fusion(iter_infos, layer_info, group_spec = fusion_group)
            comm, comp = avg_measurement(iter_infos, layer_info, fusion_group)
        else:
            comm = sim_comm
            comp = sim_comp

        print("sim comm:",[e[1] - e[0] for e in sim_comm])
        print("sim comp:", [e[1] - e[0] for e in sim_comp])

        if False:
            result_file = "./fitting_data/perf_optb32"
            iter_infos, layer_info = get_layer_wise_info(result_file)
            # fusion_iters, _ = tensor_fusion(iter_infos, layer_info, group_spec = fusion_group)
            opt_comm, opt_comp = avg_measurement(iter_infos, layer_info, fusion_group)
            compare_by_plot(opt_comm, opt_comp, comm, comp)
        else:
            compare_by_plot(sim_comm, sim_comp, comm, comp)

    # overall time
    return total_time

from model import dl_model, dl_model_data, cluster_model
#cluster_model.THREAD_OPTIONS = [64, 128, 256, 512, 1024, 2048, 4096]
def dp_optimize(model_name, batch_size, fusion_group): # dp optimization with fixed tensor group
    # granularity
    Z = 50
    # construct model
    model_info = load_model(model_name, batch_size)
    grouped_model_info = group_tensor(model_info, fusion_group)
    pids = list(grouped_model_info.keys())
    pids.sort()
    #print(pids)
    comp_time = [grouped_model_info[pid][1] for pid in pids]
    comm_size = [grouped_model_info[pid][0] for pid in pids]
    model_data = dl_model_data()
    model_data.layer_size = comm_size
    model_data.layer_computation_time = comp_time
    model = dl_model(model_data=model_data)

    # results and params
    thread_allocation = []
    opt_time = -1
    layer_num = model.layer_num

    # init dp states
    model.set_discrete_comp_time(Z)
    time_chunk = model.get_discrete_comp_time()
    states = [[[np.inf, [-1, -1], -1] for i in range(Z + 1)] for j in range(layer_num + 1)]  # [opt_value, last_hop, thread_allocation]
    for j in range(Z + 1):
        states[0][j] = [sum(time_chunk[:j]), [-1, -1], -1]

    # using a table to keep tabs of the overlap cases to reduce computation time
    overlap_cases = {}  # k - num of compute blocks - threads => [comm time, comp time]
    for k in range(1, layer_num + 1):
        comm_exceeded = [0 for i in range(len(cluster_model.THREAD_OPTIONS))]
        print("\nLayer:{}\n".format(k))
        group_id = k - 1
        print(model.get_cell_comp_index(group_id))
        # fill the overlap case table
        overlap_cases[k] = {}
        comm_task_size = model.layer_size[group_id]
        for n in range(Z + 1):
            print("Filling table k{}-n{}".format(k, n))
            overlap_cases[k][n] = {}
            comp_task_time = n * time_chunk[0] # unit time of each comp slot
            # if comm time is already shorter than comp time, we can get the time from deduct
            if sum(comm_exceeded) < len(cluster_model.THREAD_OPTIONS):
                inputs = []
                for thd in cluster_model.THREAD_OPTIONS:
                    inputs.append([comm_task_size, comp_task_time, thd])
                comm_time_s, comp_time_s = predict_func_group(inputs, type="both")
                #comm_time_s = predict_func_group(inputs, type="comm")
                #comp_time_s = predict_func_group(inputs, type="comp")
            for ri in range(len(cluster_model.THREAD_OPTIONS)):
                if comm_exceeded[ri] == 1: # comp time is already longer than the communication time
                    comm_deduct = overlap_cases[k][n - 1][cluster_model.THREAD_OPTIONS[ri]][0]
                    comp_deduct = overlap_cases[k][n - 1][cluster_model.THREAD_OPTIONS[ri]][1] + time_chunk[0]
                    overlap_cases[k][n][cluster_model.THREAD_OPTIONS[ri]] = [comm_deduct, comp_deduct]
                else:
                    overlap_cases[k][n][cluster_model.THREAD_OPTIONS[ri]] = [comm_time_s[ri], comp_time_s[ri]]
                    if comm_time_s[ri] <= comp_time_s[ri]:
                        comm_exceeded[ri] = 1
                print("Thread:{} - {}".format(cluster_model.THREAD_OPTIONS[ri], overlap_cases[k][n][cluster_model.THREAD_OPTIONS[ri]]))

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
                    ''''
                    comp_task_time = sum(time_chunk[n_hat_chunk_id + 1: comp_chunk_id + 1])
                    comm_task_size = model.layer_size[group_id]
                    # predict with blackbox function
                    comm_time = predict_func(comm_task_size, comp_task_time, thd, type="comm")
                    comp_time = predict_func(comm_task_size, comp_task_time, thd, type="comp")
                    '''
                    # just do a lookup in the table
                    comm_time = overlap_cases[k][comp_chunk_id - n_hat_chunk_id][thd][0]
                    comp_time = overlap_cases[k][comp_chunk_id - n_hat_chunk_id][thd][1]

                    # constraint
                    '''
                    if k == layer_num and n == Z:
                        print("thd: {}, comm_time: {}".format(thd, comm_time))
                    '''
                    if comm_time > comp_time and n != Z: continue
                    tmp_value += max([comm_time, comp_time])
                    if tmp_value <= opt_value:
                        opt_value = tmp_value
                        opt_last_hop = [k - 1, n_hat]
                        opt_thread_alloc = thd
                        #print("Opt value: k{}-n{}  last_hop:{}-{} add value:{} comm:{} comp:{}".format(k, n, opt_last_hop, states[k - 1][n_hat][0], max([comm_time, comp_time]), comm_time, comp_time))
            # print(opt_value)
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

    print("comm start time:", comm_start_tp)
    # validate with perf model
    print("Perf model validation---------------")
    # thread_allocation[-1] = 2048
    perf_total_time = perf_validate(model_name, batch_size, fusion_group, thread_allocation, measurement_file_mark = None)
    print("total time:{}".format(perf_total_time))
    print("------------------------------------")
    print("Thread allocation: {}".format(thread_allocation))
    return thread_allocation, opt_time


#
overlap_table = {}
def _wrapped_overlap(comm_size, comp_size, thread_num, still_time):
    # check if already in table
    comm_size = np.round(comm_size, 4)
    comp_size = np.round(comp_size, 4)
    if not (comm_size in overlap_table.keys() and comp_size in overlap_table[comm_size].keys()):
        #print("Not hit!")
        inputs = []
        for thd in cluster_model.THREAD_OPTIONS:
            inputs.append([comm_size, comp_size, thd])
        comm_time_s, comp_time_s = predict_func_group(inputs, type="both", const_time = still_time)
        #comm_time_s = predict_func_group(inputs, type="comm")
        #comp_time_s = predict_func_group(inputs, type="comp")
        if comm_size not in overlap_table.keys(): overlap_table[comm_size] = {}
        if comp_size not in overlap_table[comm_size].keys(): overlap_table[comm_size][comp_size] = {}
        for ti, thd in enumerate(cluster_model.THREAD_OPTIONS):
            overlap_table[comm_size][comp_size][thd] = [comm_time_s[ti], comp_time_s[ti]]

    if len(overlap_table[comm_size][comp_size][thread_num]) < 2:
        print( overlap_table[comm_size][comp_size][thread_num])
    #print("comm time: {}   comp time:{}".format(overlap_table[comm_size][comp_size][thread_num][0], overlap_table[comm_size][comp_size][thread_num][1]))
    return overlap_table[comm_size][comp_size][thread_num][0], overlap_table[comm_size][comp_size][thread_num][1]

def dp_cooptimize(model_name, batch_size, fusion_group=None):
    # granularity
    Z = 100
    MAX_FUSION_NUM  = 10
    latency = cluster_model.cluster_comm_lat() #0.1 # ms
    # construct model
    model_info = load_model(model_name, batch_size)
    if fusion_group is not None: grouped_model_info = group_tensor(model_info, fusion_group)
    else: grouped_model_info = model_info
    pids = list(grouped_model_info.keys())
    pids.sort()
    # print(pids)
    comp_time = [grouped_model_info[pid][1] for pid in pids]
    comm_size = [grouped_model_info[pid][0] for pid in pids]
    model_data = dl_model_data()
    model_data.layer_size = comm_size
    model_data.layer_computation_time = comp_time
    model = dl_model(model_data=model_data)

    # results and params
    thread_allocation = []
    opt_time = -1
    layer_num = model.layer_num

    # init dp states
    model.set_discrete_comp_time(Z)
    time_chunk = model.get_discrete_comp_time()
    states = [[[[np.inf, [-1, -1, -1], -1] for i in range(Z + 1)] for j in
               range(layer_num + 1)] for p in range(MAX_FUSION_NUM + 1)]  # [opt_value, last_hop, thread_allocation]
    for j in range(Z + 1):
        states[0][0][j] = [sum(time_chunk[:j]), [-1, -1, -1], -1]

    # using a table to keep tabs of the overlap cases to reduce computation time
    #overlap_case_table = {}  # k - num of compute blocks - threads => [comm time, comp time]
    for s in range(1, MAX_FUSION_NUM + 1):
        print("Stage:{}".format(s))
        for l in range(s, layer_num + 1):  # l >= s
            layer_id = l - 1
            # print(model.get_cell_comp_index(layer_id))
            for z in range(Z + 1):
                comp_chunk_id = z - 1
                start_chunk_id = model.get_cell_comp_index(layer_id)
                z_hat_start = start_chunk_id + 1
                opt_value = np.inf
                opt_last_hop = [-1, -1, -1]
                opt_thread_alloc = -1
                # start searching
                if l == layer_num:
                    ll = l + 1  # to avoid empty group in the middle
                else:
                    ll = l
                for l_hat in range(ll):
                    gradient_size = 0
                    for li in range(l_hat + 1, l + 1):
                        gradient_size += model.layer_size[li - 1]
                    for z_hat in range(z_hat_start, z + 1):
                        z_hat_chunk_id = z_hat - 1
                        comp_size = sum(time_chunk[z_hat_chunk_id + 1: comp_chunk_id + 1])
                        for thd in cluster_model.THREAD_OPTIONS:
                            tmp_value = states[s - 1][l_hat][z_hat][0]
                            comm_time, comp_time = _wrapped_overlap(gradient_size, comp_size, thd)
                            # constraint
                            '''
                            if k == layer_num and n == Z:
                                print("thd: {}, comm_time: {}".format(thd, comm_time))
                            '''
                            if comm_time > comp_time and z != Z: continue
                            tmp_value += max([comm_time, comp_time])
                            if gradient_size > 0: tmp_value += latency
                            if tmp_value < opt_value:
                                # print(1)
                                opt_value = tmp_value
                                opt_last_hop = [s - 1, l_hat, z_hat]
                                opt_thread_alloc = thd
                                # if s == 3: exit()
                                # print(comp_size)
                                # print("Start id:{} thd:{} sum comp:{} ".format(z_hat_start, thd, len(time_chunk)))
                                # print("Opt value:{}  s{}-l{}-z{}  last_hop:{} add value:{} comm:{} comp:{}".format(opt_value, s, l, z, opt_last_hop, max([comm_time,comp_time]),comm_time,comp_time))
                                # if s == 3 and l == 15: exit()
                # print(opt_value)
                states[s][l][z] = [opt_value, opt_last_hop, opt_thread_alloc]

    opt_res = states[MAX_FUSION_NUM][layer_num][Z]
    print("opt:", opt_res)
    opt_time = opt_res[0]
    # exit()

    for li in range(len(states)):
        print(states[li])
    # gather thread allocation
    comm_start_tp = []
    layer_fusion = []
    trace_state = opt_res
    last_hop = trace_state[1]
    tmp_li = layer_num
    while last_hop[0] != -1:
        #print("xxxx:", last_hop)
        #print(trace_state)
        layer_fusion.insert(0, tmp_li - last_hop[1])
        tmp_li = last_hop[1]
        thread_allocation.insert(0, trace_state[-1])
        trace_state = states[last_hop[0]][last_hop[1]][last_hop[2]]
        last_hop = trace_state[1]
        comm_start_tp.insert(0, trace_state[0])

    print("layer_fusion:", layer_fusion)
    assert sum(layer_fusion) == layer_num
    print("total comp (ms):", sum(model.layer_computation_time))
    print("comm start time:", comm_start_tp)
    # validate with perf model
    model.group(layer_fusion)
    print("Perf model validation---------------")
    perf_total_time = perf_validate(model_name, batch_size, fusion_group, thread_allocation, measurement_file_mark=None)
    print("total time:{}".format(perf_total_time))
    print("------------------------------------")
    print("Thread allocation: {}".format(thread_allocation))

    return thread_allocation, opt_time, layer_fusion


import time
if __name__ == "__main__":
    start_t = time.time()
    #preprocess()
    #exit()
    #preprocess_add_nooverlap()
    #perf_validate("VGG16", 32, [4, 4, 4, 8, 8, 4], [256 for i in range(6)], measurement_file_mark="test1122")

    '''
    perf_validate("VGG16", 32, [1, 7, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2048 for i in [1, 7, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                  measurement_file_mark="fake")
    #'''

    # our opt
    '''
    perf_validate("VGG16", 32, [25, 6, 1], #[4, 2, 3, 23]
                  [2048 for i in range(3)],
                  measurement_file_mark="fake")
    #'''
    '''
    perf_validate("VGG16", 32, [1,1,1,1,1,2,20,5],
                  [256,64,512,64,512,256,512,256],
                  measurement_file_mark="mergeb32")
    #'''

    #perf_validate("VGG16", 32, [4, 4, 4, 8, 8, 4], [2048 for i in range(6)], measurement_file_mark="test2048")
    #perf_validate("VGG16", 4, [4, 4, 4, 8, 8, 4], [2048 for i in range(6)], measurement_file_mark="test2048b4")
    #dp_optimize("VGG16", 4, [4, 4, 4, 8, 8, 4])

    #perf_validate("VGG16", 32, [7, 20, 5],  [256, 512, 4096],measurement_file_mark="tmp")
    #perf_validate("VGG16", 32, [9, 18, 5], [512, 512, 4096], measurement_file_mark="tmp2")
    perf_validate("VGG16", 16, [9, 18, 5], [512, 512, 4096], measurement_file_mark="tmp2")

    #vgg16 - b2 [1, 2, 2, 20, 7]   [512, 2048, 4096, 2048, 4096]
    #perf_validate("ResNet152", 32, [80,80,80,154,152,76], [512,512,512,512,512,4096], measurement_file_mark="resnetb32")

    # 64 results: small allocation prioritized: [512, 512, 256, 256, 256, 256] - perf: 307
    # large prioritized: [512, 512, 4096, 4096, 4096, 4096]
    #perf_validate("VGG16", 64, [4, 4, 4, 8, 8, 4],  [512,512,512,512,512,4096], measurement_file_mark="optb64-fitting")
    #512,512,256,256,256,256  [64, 512, 1024, 128, 4096, 512]
    end_t = time.time()
    print("Total time consumption: {}s".format(end_t - start_t))