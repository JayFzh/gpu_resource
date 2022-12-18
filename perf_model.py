from model import dl_model, cluster_model
from interference import overlap1 as overlap

class alloc_policy:
    def __init__(self, thd_plc):
        self.thread_allocation = thd_plc

from scripts.contention_fit2 import _fitting_comm
def overlap_perf_comm_stage(model: dl_model, plc: alloc_policy, debug = False):
    assert len(plc.thread_allocation) == model.layer_num
    layer_num = model.layer_num
    stage_num = layer_num + 1
    stage_time = [0 for i in range(stage_num)]
    comp_time = model.layer_computation_time
    #total_comp_time = sum(model.layer_computation_time)
    comm_sizes = model.layer_size

    comp_index = 0
    comp_remain = 0
    comm_remain = 0

    #statistics
    total_overlap_comp_time = 0
    total_overlap_comm_time = 0
    total_latency = 0
    comm_stats = []
    comp_stats = []

    # compute the stage time layer by layer
    lat_to_add = 0
    still_time = 0 # latency time which does not change with resource allocation
    for si in range(stage_num):
        li = si - 1
        if si == 0:
            stage_time[si] = comp_time[comp_index]
            total_overlap_comp_time += stage_time[si]
            comp_index += 1
            continue
        last_stage_end = stage_time[si - 1]
        current_stage_end = last_stage_end
        comm_remain = comm_sizes[li]
        lat_to_add = 1
        still_time = cluster_model.cluster_comm_lat()
        if comm_remain > 0: total_latency += still_time
        #account for latency
        '''
        if comm_remain > 0:
            print("xxxxx")
            current_stage_end += cluster_model.cluster_comm_lat()
            comp_remain = max(0, comp_remain - cluster_model.cluster_comm_lat())
        '''
        while comm_remain > 0:
            #print("New communication")
            if comp_remain == 0 and comp_index < layer_num:
                comp_remain = comp_time[comp_index]
                comp_index += 1
            print("!!!!si:{}, comp_remain:{}, comm remain:{} still remain:{} comp_index:{}".format(si, comp_remain, comm_remain, still_time, comp_index))
            overlap_comm, overlap_comp = overlap(comm_remain, comp_remain, plc.thread_allocation[li], still_time)
            full_overlap_comm =  _fitting_comm([comm_remain, plc.thread_allocation[li]])
            print("overlap comm:{}, overlap comp:{}".format(overlap_comm, overlap_comp))

            #[lyz] todo: more gpu resources would not speedup this part of comm
            '''
            if lat_to_add:
                lat_to_add = 0
                overlap_comm += cluster_model.cluster_comm_lat()
            '''
            if overlap_comm <= overlap_comp:
                comm_remain = 0
                # stats
                total_overlap_comm_time += overlap_comm
                still_time = 0
                # confirm stage time
                assert  comp_index >= (li + 2) or comp_index == layer_num
                if comp_index == (li + 2):
                    # wait for the comp to end
                    current_stage_end += overlap_comp
                    # stats
                    total_overlap_comp_time += overlap_comp
                    comp_remain = 0
                else:
                    # get remain comp time
                    current_stage_end += overlap_comm
                    # stats
                    total_overlap_comp_time += overlap_comm
                    unoverlapped = overlap_comp - overlap_comm
                    comp_remain = unoverlapped
            else:
                if comp_index == layer_num:
                    # stats
                    total_overlap_comm_time += overlap_comm
                    current_stage_end += overlap_comm
                    if comp_remain > 0: total_overlap_comp_time += overlap_comp
                    comp_remain = 0
                    break
                # get next comp
                current_stage_end += overlap_comp
                # stats
                total_overlap_comp_time += overlap_comp
                total_overlap_comm_time += overlap_comp
                # lyz - lat
                remaining_still = max(0, still_time - overlap_comp)
                overlapped_comm_t = max(0, overlap_comp - still_time)
                comm_remain *= (1 - overlapped_comm_t / full_overlap_comm)  # bug: here should be full-overlap_comm_time
                still_time = remaining_still
                comp_remain = 0
        stage_time[si] = current_stage_end
    if debug: print(stage_time)
    print("[perf model] total computation after overlap:{}".format(total_overlap_comp_time))
    print("[perf model] total communication after overlap:{}".format(total_overlap_comm_time))
    print("[perf model] total latency:{}".format(total_latency))
    return stage_time[-1]


def overlap_perf_comp_stage(model: dl_model, env: cluster_model, plc: alloc_policy):
    # indeed, modeling in terms of communication stages seems better
    assert len(plc.thread_allocation) == model.layer_num
    layer_num = model.layer_num
    stage_num = layer_num + 1
    t_1 = [0 for i in range(stage_num)] # comp time from last stage
    t_2 = [0 for i in range(stage_num)] # comp time completely overlapped in current stage
    t_3 = [0 for i in range(stage_num)] # comp time extend to the next stage
    comp_time = model.layer_computation_time
    comm_sizes = model.layer_size
    s1 = 0
    s2 = 0
    for k in range(1, stage_num):
        s2 += 0