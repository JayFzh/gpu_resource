import matplotlib.pyplot as plt
import numpy as np
#import queue

fzh_list_total = [[],[],[],[],[],[],[]]
detail = []
last_thread = ''

'''
apis
'''
def get_layer_wise_info(file_path):
    '''
    :param file_path:
    :return: iter_info: {pid: {op_name:[start_t, end_t]}}
    '''
    # for logging the seperated computation time point
    name_to_pid = {}
    #comp_pid_to_name = {}
    comp_pid_to_tensor_pid = {}

    comp_stats_data = {}
    layers_stats_data = {} # {iter_id: layer_wise_training_info}
    layers_info = {} # {pid: (gradient_size, comp_time (un-overlapped))}
    # intermediate data for parsing the file
    min_timestamp = np.inf
    inter_data = {} #pid: [duration event queue,]
    target_pids = set()
    comp_pids = set()
    with open(file_path, "r") as f:
        data = f.read()
        data = data[:-1]
        data += ']'
        data = eval(data)
        for entry in data:
            name = None
            if "name" in entry.keys():
                name = entry["name"]
            pid = entry["pid"]
            # get valid pids
            if name == "process_name":
                if "name" not in entry["args"].keys(): continue
                pname = entry["args"]["name"]
                if "Allreduce" in pname or "allreduce" in pname:
                    if "comptimeline" not in pname:
                        target_pids.add(pid)
                        if pid not in inter_data.keys(): inter_data[pid] = [0, []]
                        if pid not in layers_info.keys(): layers_info[pid] = [0, 0] # gradient size, comp time
                        comp_name = pname+"_comptimeline"
                        if comp_name in name_to_pid.keys():
                            comp_pid = name_to_pid[comp_name]
                            # add to query table
                            comp_pid_to_tensor_pid[comp_pid] = pid
                        else:
                            name_to_pid[pname] = pid
                    else:
                        comp_pids.add(pid)
                        if pid not in inter_data.keys(): inter_data[pid] = [0, []]
                        tensor_name = pname[:-len("_comptimeline")]
                        if tensor_name in name_to_pid.keys():
                            tensor_pid = name_to_pid[tensor_name]
                            # add to query table
                            comp_pid_to_tensor_pid[pid] = tensor_pid
                        else:
                            name_to_pid[pname] = pid

            if pid not in target_pids and pid not in comp_pids: continue
            etype = entry["ph"]
            if name is None: assert etype == "E"
            if etype == "M": continue
            elif etype == "B":
                start_t = entry["ts"] / 1000.0 # ms
                if start_t < min_timestamp: min_timestamp = start_t
                inter_data[pid][1].append((name, start_t))
            elif etype == "X": # comp currently does not include any X
                start_t = entry["ts"] / 1000.0 # ms
                if start_t < min_timestamp: min_timestamp = start_t
                duration = entry["dur"] / 1000.0 # ms
                end_t = start_t + duration
                iter_id = inter_data[pid][0]
                if iter_id not in layers_stats_data.keys():
                    layers_stats_data[iter_id] = {}
                    # print("New iter:{}".format(iter_id))
                if pid not in layers_stats_data[iter_id].keys():
                    layers_stats_data[iter_id][pid] = {}
                layers_stats_data[iter_id][pid][name] = [start_t, end_t, duration]
            elif etype == "E":
                end_t = entry["ts"] / 1000.0
                if len(inter_data[pid][1]) == 0:
                    assert pid in comp_pids
                    print("NO beginning: {}".format(pid))
                    start_t = end_t
                    name = "WAIT_FOR_DATA"
                else:
                    start_info = inter_data[pid][1].pop()
                    #print(start_info)
                    start_t = start_info[1]
                    name = start_info[0]
                duration = end_t - start_t
                # the WAIT_FOR_DATA is obtained from the seperate comp timeline
                if name == "WAIT_FOR_DATA" and pid in target_pids:
                    continue
                if pid in comp_pids:
                    assert name == "WAIT_FOR_DATA"
                    #pid = comp_pid_to_tensor_pid[pid]
                    iter_id = inter_data[pid][0]
                    if iter_id not in comp_stats_data.keys():
                        comp_stats_data[iter_id] = {}
                        # print("New comp iter:{}".format(iter_id))
                    if pid not in comp_stats_data[iter_id].keys():
                        comp_stats_data[iter_id][pid] = {}
                    comp_stats_data[iter_id][pid][name] = [start_t, end_t, duration]
                    inter_data[pid][0] += 1
                    continue

                iter_id = inter_data[pid][0]
                if iter_id not in layers_stats_data.keys():
                    layers_stats_data[iter_id] = {}
                    # print("New iter:{}".format(iter_id))
                if pid not in layers_stats_data[iter_id].keys():
                    layers_stats_data[iter_id][pid] = {}
                layers_stats_data[iter_id][pid][name] = [start_t, end_t, duration]
                if name == "ALLREDUCE": # the end of an iteration
                    inter_data[pid][0] += 1
                    #print(entry)
                    if "args" in entry.keys():
                        #assert "args" in entry.keys()
                        shape =  eval(entry["args"]["shape"])#MB
                        size = 1
                        for s in shape: size *= s
                        size = size * 4 / 1024 / 1024
                        layers_info[pid][0] = size

    #second pass: replace WAIT_FOR_DATA
    to_del = []
    for iter_id in comp_stats_data.keys():
        if iter_id not in layers_stats_data.keys():
            continue
        if len(comp_stats_data[iter_id].keys()) != len(layers_stats_data[iter_id].keys()):
            to_del.append(iter_id)
            print("comp and tensor pid not equal in length!!!")
            continue
        for comp_pid in comp_stats_data[iter_id].keys():
            name = "WAIT_FOR_DATA"
            tensor_pid = comp_pid_to_tensor_pid[comp_pid]
            layers_stats_data[iter_id][tensor_pid][name] = comp_stats_data[iter_id][comp_pid][name]
            '''
            print(layers_stats_data[iter_id].keys())
            print(comp_stats_data[iter_id].keys())
            print(len(layers_stats_data[iter_id].keys()), len(comp_stats_data[iter_id].keys()))
            print(iter_id)
            '''
    for iter_id in to_del:
        layers_stats_data.__delitem__(iter_id)

    #third pass: generate relative timestamps
    for iter_id in layers_stats_data.keys():
        for pid in layers_stats_data[iter_id].keys():
            for name in layers_stats_data[iter_id][pid].keys():
                layers_stats_data[iter_id][pid][name][0] -= min_timestamp
                layers_stats_data[iter_id][pid][name][1] -= min_timestamp

    # print(layers_stats_data.keys())
    # change all pid to consecutive numbers - indexes start with 1
    pids = list(layers_stats_data[0].keys())
    pids.sort()
    new_layers_stats_data = {}
    new_layers_info = {}
    for iter_id in layers_stats_data.keys():
        if iter_id < 1: continue
        new_layers_stats_data[iter_id] = {}
        for pid in layers_stats_data[iter_id].keys():
            new_pid = pids.index(pid) + 1
            new_layers_info[new_pid] = layers_info[pid]
            new_layers_stats_data[iter_id][new_pid] = layers_stats_data[iter_id][pid]
    layers_stats_data = new_layers_stats_data
    layers_info = new_layers_info

    return layers_stats_data, layers_info

'''
tensor fusion
'''
def tensor_fusion(iter_info, layer_info, group_spec = None):
    if group_spec: return _designated_fusion(iter_info, layer_info, group_spec)
    else: return _auto_fusion(iter_info, layer_info)

def _auto_fusion(iter_info, layer_info):
    # determine fusion group
    pids = list(iter_info.keys())
    pids.sort()
    fusion_group = []
    group_size = 0
    last_allreduce_start = -1
    last_allreduce_end = -1
    for i, pid in enumerate(pids):
        allreduce_start = iter_info[pid]["NCCL_ALLREDUCE"][0]
        allreduce_end = iter_info[pid]["NCCL_ALLREDUCE"][1]
        if allreduce_start >= last_allreduce_end:
            if group_size > 0: fusion_group.append(group_size)
            group_size = 1
        else:
            group_size += 1
        last_allreduce_start = allreduce_start
        last_allreduce_end = allreduce_end
    if group_size > 0:
        fusion_group.append(group_size)
    print("fusion group:", fusion_group)

    return _designated_fusion(iter_info, layer_info, fusion_group)

def _designated_fusion(iter_info, layer_info, group_spec):
    fused_iter_info = {}
    fusion_info = {}
    pids = list(iter_info.keys())
    pids.sort()
    p_count = 0
    fusion_id = 0
    fusion_info_entry = np.array([0 for i in range(len(layer_info[pids[0]]))])
    for i, pid in enumerate(pids):
        p_count += 1
        fusion_info_entry = fusion_info_entry + layer_info[pid]
        if p_count == group_spec[fusion_id]:
            fusion_info[fusion_id] = list(fusion_info_entry)
            start_pid = pids[i + 1 - p_count]
            start_id = i + 1 - p_count
            fused_iter_info[fusion_id] = iter_info[start_pid] # fused_iter_info is the beginning of the fusion group info
            # fix: recalculate the WAIT_FOR_DATA time point
            end_wfd_time = iter_info[pid]["WAIT_FOR_DATA"][1] # only the end time is accurate
            if start_id == 0:
                start_wfd_time = iter_info[start_pid]["WAIT_FOR_DATA"][1]
            else:
                start_wfd_time = iter_info[pids[start_id - 1]]["WAIT_FOR_DATA"][1]
            fused_iter_info[fusion_id]["WAIT_FOR_DATA"][0] = start_wfd_time
            fused_iter_info[fusion_id]["WAIT_FOR_DATA"][1] = end_wfd_time
            fused_iter_info[fusion_id]["LAST_NCCL_ALLREDUCE"] = list()
            fused_iter_info[fusion_id]["LAST_NCCL_ALLREDUCE"].append(iter_info[pid]["NCCL_ALLREDUCE"][0])
            fused_iter_info[fusion_id]["LAST_NCCL_ALLREDUCE"].append(iter_info[pid]["NCCL_ALLREDUCE"][1])
            '''
            if p_count > 1: assert "MEMCPY_IN_FUSION_BUFFER" in iter_info[pid].keys()
            else:
                if "MEMCPY_IN_FUSION_BUFFER" in iter_info[pid].keys():
                    print("Group spec:", group_spec)
                    print_timeline(iter_info, layer_info)
                assert "MEMCPY_IN_FUSION_BUFFER" not in iter_info[pid].keys()
            '''
            fusion_id += 1
            p_count = 0
            fusion_info_entry = np.array([0 for i in range(len(layer_info[pids[0]]))])
            
    # print(fused_iter_info) 
    # fused_iter_info is the result of iter_info's fusion 
    # print(fusion_info) 
    # fusion_info is the result of model_info's fusion 
    # fusion_info[pid][0] is size
    # fusion_info[pid][1] is the comp time
    # fusion_info[pid][2] is the std time
    
    return fused_iter_info, fusion_info

'''
computation time extraction
'''
def fill_layer_computation(layer_stats, layer_info):
    inter_data = {}
    res = {}
    for i in layer_info.keys():
        inter_data[i] = []
        res[i] = [layer_info[i][0], 0, 0] # pid: [grad size, comp time, comp time std]
    pids = list(layer_info.keys())
    pids.sort()
    for iter_id in layer_stats.keys():
        if iter_id == 0: continue
        iter_info = layer_stats[iter_id]
        last_comp_end = -1
        for pid in pids:
            comp_end = iter_info[pid]["WAIT_FOR_DATA"][1]
            if last_comp_end < 0:
                comp_time = 0
            else:
                comp_time = comp_end - last_comp_end
            last_comp_end = comp_end
            inter_data[pid].append(comp_time) # in ms
    for pid in pids:
        res[pid][1] = max(0, np.mean(inter_data[pid]))
        res[pid][2] = max(0, np.std(inter_data[pid]))

    # debug
    print()
    total = []
    for pid in pids:
        print("PID: {:<10d}\t{:<10.4f}MB\tcomp:{:<10.4f}ms\tstd:{:<10.4f}ms".format(pid, res[pid][0], res[pid][1], res[pid][2]))
        total.append([res[pid][0], res[pid][1], res[pid][2]])
    total = np.sum(total, axis=0)
    print("Total\t{:<10.4f}MB\tcomp:{:<10.4f}ms\tstd:{:<10.4f}ms".format(total[0], total[1],total[2]))
    print()
    global fzh_list_total
    fzh_list_total[0].append(total[1])
    global detail
    detail.append(total[1])
    return res

'''
communication time extraction
'''
def fill_layer_communication(layer_stats, layer_info,fusion,thread):
    global last_thread
    # if last_thread != thread:
    #     with open('statistics.txt','a') as f:
    #         f.write(thread+'\n') 
    #     with open('detail2.txt','a') as f:
    #         f.write(thread+'\n') 
    #     last_thread = thread
    fusion = fusion.split(',')
    fusion = [int(i) for i in fusion]
    for i,_ in enumerate(fusion):
        if i==0:
            continue
        fusion[i] = fusion[i]+fusion[i-1]
    fusion.insert(0,1)
    fusion.pop()
    inter_data = {}
    res = {}
    for i in layer_info.keys():
        inter_data[i] = []
        res[i] = [layer_info[i][0], 0, 0] # pid: [grad size, comm time, comm time std]
    pids = list(layer_info.keys())
    pids.sort()
    for iter_id in layer_stats.keys():
        if iter_id == 0: continue
        iter_info = layer_stats[iter_id]
        for pid in fusion:
            comm_start = iter_info[pid]["NCCL_ALLREDUCE"][0]
            comm_end = iter_info[pid]["NCCL_ALLREDUCE"][1]
            inter_data[pid].append(comm_end-comm_start) # in ms
    for pid in fusion:
        res[pid][1] = max(0, np.mean(inter_data[pid]))
        res[pid][2] = max(0, np.std(inter_data[pid]))

    global fzh_list_total,detail

    # debug
    print()
    total = []
    i = 0
    for pid in pids:
        if pid in fusion:
            print("PID: {:<10d}\t{:<10.4f}MB\tcomm:{:<10.4f}ms\tstd:{:<10.4f}ms".format(pid, res[pid][0], res[pid][1], res[pid][2]))
            fzh_list_total[i+1].append(res[pid][1])
            detail.append(res[pid][1])
            i += 1
        total.append([res[pid][0], res[pid][1], res[pid][2]])
    total = np.sum(total, axis=0)
    print("Total\t{:<10.4f}MB\tcomm:{:<10.4f}ms\tstd:{:<10.4f}ms".format(total[0], total[1],total[2]))
    print()
    detail.append(total[1])
    fzh_list_total[6].append(total[1])
    # if len(fzh_list_total[0]) == 10:
    #     ans = []
    #     for i,_ in enumerate(fzh_list_total):
    #         ans.append(np.mean(fzh_list_total[i]))
    #     with open('statistics.txt','a') as f:
    #         for i in ans:
    #             f.write(str(i)+' ')
    #         f.write('\n')
    #     for _,i in enumerate(fzh_list_total):
    #         i.clear()
    # with open('detail2.txt','a') as f:
    #     for i in detail:
    #         f.write(str(i)+' ')
    #     f.write('\n')
    detail.clear()
    return res

def is_overlap(a1,b1,a2,b2):
    if (b1>a2 and b1<b2) or (a1<a2 and b1>b2) or (a1>a2 and a1<b2) :
        return True
    else:
        return False


'''
contention data entraction
'''
def extract_contention_data(iter_info):
    # get the overlap relation of all fusions and see if it is one to one
    comm_overlap = {}
    comp_overlap = {}
    pids = list(iter_info.keys()) # todo: skip tensor 0 or not?
    pids.sort()
    comm_time = {}
    comp_time = {}
    multi_comp_overlap = {}
    multi_comm_overlap = {}
    for i, pid in enumerate(pids):
        comm_overlap[pid] = []
        comp_overlap[pid] = []
        multi_comp_overlap[pid] = []
        multi_comm_overlap[pid] = []
        comm_time[pid] = [iter_info[pid]["NCCL_ALLREDUCE"][0], iter_info[pid]["NCCL_ALLREDUCE"][1]]
        #if i > 0: comp_time[pid] = [iter_info[pid]["WAIT_FOR_DATA"][0], iter_info[pid]["WAIT_FOR_DATA"][1]]
        #else:
        comp_time[pid] =  [iter_info[pid]["WAIT_FOR_DATA"][0], iter_info[pid]["WAIT_FOR_DATA"][1]]
        
    
    # fzh-overlap
    
    for id in pids:
        
        comp_period = comp_time[id]
        comm_period = comm_time[id]
        for pid in range(0,id):
            if is_overlap(comp_period[0],comp_period[1],comm_time[pid][0],comm_time[pid][1]):
                multi_comp_overlap[id].append(pid)
        if len(multi_comp_overlap[id])>1:
            for index in range(0,len(multi_comp_overlap[id])-1):
                pid = multi_comp_overlap[id][index]
                next = index + 1
                while next < len(multi_comp_overlap[id]):
                    next_pid = multi_comp_overlap[id][next] 
                    if is_overlap(comm_time[pid][0],comm_time[pid][1],comm_time[next_pid][0],comm_time[next_pid][1]):
                        multi_comm_overlap[id].append((pid,next_pid))
                    next += 1
    comp_pairs = []
    for id in pids:
        comp_period = comp_time[id]
        comm_period = comm_time[id]
        for pid in multi_comp_overlap[id]:
            if comm_time[pid][0] <= comp_period[0] and comm_time[pid][1] >= comp_period[1]:
                flag = False
                for pair in comp_pairs:
                    if id in pair["comp_ids"]:
                        flag = True
                        pair["comm_ids"].append(pid)
                        pair["comm_time"].append(comm_time[pid][1]-comm_time[pid][0])
                if flag == False:
                    entry = {}
                    entry["comp_ids"] = [id]
                    entry["comm_ids"] = [pid]
                    entry["comm_time"] = [comm_time[pid][1]-comm_time[pid][0]]
                    entry["comp_time"] = comp_period[1]-comp_period[0]
                    comp_pairs.append(entry)

    print()
    print('multiple overlap:')
    for id in pids:
        print('comp pid',id,'overlap with comm pid',multi_comp_overlap[id],'\ncomm overlap:',multi_comm_overlap[id])
        
    for id in pids:
        comp_period = comp_time[id]
        comm_period = comm_time[id]
        for pid in pids:
            if comm_time[pid][0] > comp_period[0] and comm_time[pid][0] < comp_period[1] \
                    or comm_time[pid][1] > comp_period[0] and comm_time[pid][1] < comp_period[1]\
                    or comm_time[pid][0] <= comp_period[0] and comm_time[pid][1] >= comp_period[1] :
                comp_overlap[id].append(pid)
            if comp_time[pid][0] > comm_period[0] and comp_time[pid][0] < comm_period[1] \
                    or comp_time[pid][1] > comm_period[0] and comp_time[pid][1] < comm_period[1]\
                    or comp_time[pid][0] <= comm_period[0] and comp_time[pid][1] >= comm_period[1] :
                comm_overlap[id].append(pid)

    overlap_pairs = []
    # one-to-one or many-to-one ?
    for pid in pids:
        #comm overlap
        overlap_comp_ids = comm_overlap[pid]
        fit = len(overlap_comp_ids) > 0
        comp_time = 0
        comm_time = iter_info[pid]["NCCL_ALLREDUCE"][1] - iter_info[pid]["NCCL_ALLREDUCE"][0]
        for id in overlap_comp_ids:
            comp_time += (iter_info[id]["WAIT_FOR_DATA"][1] - iter_info[id]["WAIT_FOR_DATA"][0])
            if len(comp_overlap[id]) > 1:
                fit = False
                break
        if fit:
            entry = {}
            entry["comm_ids"] = [pid]
            entry["comp_ids"] = overlap_comp_ids
            entry["comm_time"] = comm_time
            entry["comp_time"] = comp_time
            overlap_pairs.append(entry)

        # comp overlap
        overlap_comm_ids = comp_overlap[pid]
        fit = len(overlap_comm_ids) > 0
        comp_time = iter_info[pid]["WAIT_FOR_DATA"][1] - iter_info[pid]["WAIT_FOR_DATA"][0]
        comm_time = 0
        for id in overlap_comm_ids:
            comm_time += (iter_info[id]["NCCL_ALLREDUCE"][1] - iter_info[id]["NCCL_ALLREDUCE"][0])
            if len(comm_overlap[id]) > 1:
                fit = False
                break
        if fit:
            entry = {}
            entry["comp_ids"] = [pid]
            entry["comm_ids"] = overlap_comm_ids
            entry["comm_time"] = comm_time
            entry["comp_time"] = comp_time
            # make sure it is not repeatedly added
            repeated = False
            for pair in overlap_pairs:
                if pair["comp_ids"] == entry["comp_ids"] and pair["comm_ids"] == entry["comm_ids"]:
                    repeated = True
                    break
            if not repeated: overlap_pairs.append(entry)

        # computation

    '''
    for pid in pids:
        if len(comm_overlap[pid]) == 1:
            comp_id = comm_overlap[pid][0]
            assert pid in comp_overlap[comp_id]
            if len(comp_overlap[comp_id]) == 1:
                overlap_pairs.append([pid, comp_id])

        # debug disp
        print("ID: {}".format(pid))
        print("comm overlap: {}".format(comm_overlap[pid]))
        print("comp overlap: {}".format(comp_overlap[pid]))
    '''
    print("single overlap pair:", overlap_pairs)
    #print(pids)
    #exit()
    return overlap_pairs,comp_pairs

'''
get communication without overlap
'''
def extract_unoverlap_communication(iter_info, layer_info, thread_allocation):
    # get the communication info of the last layer
    pids = list(iter_info.keys())
    pids.sort()
    last_pid = pids[-1]
    comm_size = layer_info[last_pid][0]
    comm_time = iter_info[last_pid]["NCCL_ALLREDUCE"][1] - iter_info[last_pid]["NCCL_ALLREDUCE"][0]
    unoverlap_comms = [comm_time, 0, comm_size, 0, thread_allocation]

    return unoverlap_comms



'''
plots
'''
def print_timeline(iter_info, layer_info):
    #iter_info = layer_stats[iter_num]
    pids = list(iter_info.keys())
    pids.sort()
    for i, pid in enumerate(pids):
        #if i== 0: print("PID: {}\t\t\t{}MB\t\t\t{}ms\t\t\t{}ms".format(pid, layer_info[pid][0], layer_info[pid][1], iter_info[pid]["WAIT_FOR_OTHER_TENSOR_DATA"][1] - iter_info[pid]["WAIT_FOR_OTHER_TENSOR_DATA"][0]))
        #else: print("PID: {}\t\t\t{}MB\t\t\t{}ms\t\t\t{}ms".format(pid, layer_info[pid][0], layer_info[pid][1], iter_info[pid]["WAIT_FOR_OTHER_TENSOR_DATA"][1] - iter_info[pid]["WAIT_FOR_DATA"][0]))
        print("PID: {}\t\t\t{}MB\t\t\t{}ms\t\t\t{}ms".format(pid, layer_info[pid][0], layer_info[pid][1],
                                                             iter_info[pid]["WAIT_FOR_DATA"][1] -
                                                             iter_info[pid]["WAIT_FOR_DATA"][0]))
        names = list(iter_info[pid].keys())
        start_ts = [iter_info[pid][name][0] for name in names]
        end_ts = [iter_info[pid][name][1] for name in names]
        sort_id = np.argsort(start_ts)
        #print(names)
        names = np.array(names)[sort_id]
        start_ts = np.array(start_ts)[sort_id]
        end_ts = np.array(end_ts)[sort_id]
        for i, name in enumerate(names):
            print("{:<40s}\t{:<10.4f}\t{:<10.4f}".format(name, start_ts[i], end_ts[i]))
        print()

def plot_timeline(layer_stats):
    pass


# test
if __name__ == "__main__":
    file = "../scripts/debug/Bert_debug"
    layer_stats, layer_info = get_layer_wise_info(file)
    _ = fill_layer_computation(layer_stats, layer_info)
    exit()
    file_path = "../scripts/fitting_data/perf_test1122"
    layer_stats, layer_info  = get_layer_wise_info(file_path)
    print_timeline(layer_stats[6], layer_info)
    fusion_iter, fusion_info = tensor_fusion(layer_stats[6], layer_info)
    print_timeline(fusion_iter, fusion_info)
    extract_contention_data(fusion_iter)

if __name__ == "__main__x":
    #file = "../scripts/data3/VGG16_b4_t2048_g0"
    #_, _ = get_layer_wise_info(file)
    #exit()
    file_path = "../scripts/data3/VGG16_b4_t2048_g0"
    layer_stats, layer_info  = get_layer_wise_info(file_path)
    print_timeline(layer_stats[6], layer_info)
    fusion_iter, fusion_info = tensor_fusion(layer_stats[6], layer_info)
    print_timeline(fusion_iter, fusion_info)
    #extract_contention_data(fusion_iter)
    alloc = extract_unoverlap_communication(fusion_iter, fusion_info, 0)
    print(alloc)

