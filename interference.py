from model import cluster_model, dl_model
from typing import List
from scripts.run_data_fit import _wrapped_overlap

def overlap_test(start_comp_chunk_id, end_comp_chunk_id, group_id, thread_num, model: dl_model, still_time):
    time_chunk = model.get_discrete_comp_time()
    comp_time = sum(time_chunk[start_comp_chunk_id: end_comp_chunk_id + 1])
    gradient_size = model.layer_size[group_id]
    transfer_time = gradient_size / cluster_model.bandwidth * 1000.0 #ms

    if gradient_size == 0: still_time = 0

    #overlap_comm_time = transfer_time
    #overlap_comp_time = comp_time

    overlap_comm_time, overlap_comp_time = overlap1(gradient_size, comp_time, thread_num, still_time)

    if start_comp_chunk_id > end_comp_chunk_id: return overlap_comm_time, 0
    return overlap_comm_time, overlap_comp_time

def overlap_test2(comp_time, gradient_size, thread_num, model: dl_model, still_time):

    overlap_comm_time, overlap_comp_time = overlap1(gradient_size, comp_time, thread_num, still_time)
    if comp_time == 0:
        assert overlap_comp_time == 0
        overlap_comp_time = 0
    if gradient_size == 0:
        assert overlap_comm_time == 0
        overlap_comm_time = 0

    return overlap_comm_time, overlap_comp_time

def overlap1_small(comm_size, comp_time, comm_thd): # MB, ms
    # model setting
    # '''  communication time  '''
    # 50Gb - 1024 thread - 50 * 1024 / 8 MB - 1s
    # \delta * size / (alpha)^(k) / thread num = comm time
    # => 0.05 * 6400 / (0.95) ** (1024 / 64) / 1024 * 1000 =
    # '''  computation time  '''
    # increase: 256 * 70 * \omega = 7 ms
    # => 0.000390625 * 256 * 70 =

    max_thd_usage = min(2872, comm_thd)
    comm_time1 = 0.05 * comm_size / (0.95) ** (max_thd_usage / 64) / max_thd_usage * 1000.0
    #print("comm1: {}".format(comm_time1))
    comm_time2 = comm_size / cluster_model.bandwidth * 1000.0 # a simple estimation
    comm_time = max([comm_time1, comm_time2])
    overlapped_comp_time = comp_time + 0.000390625 * comm_thd * comm_time
    if comm_time > overlapped_comp_time:
        # comm_thd * 0.000390625 * x + comp_time = x
        # => x = comp_time / (1 - comm_thd * 0.000390625)
        assert (1 - comm_thd * 0.000390625) > 0
        overlapped_comp_time = comp_time / (1 - comm_thd * 0.000390625)

    if comp_time == 0: overlapped_comp_time = 0

    #print("thd:{}, comm:{} , comp:{} ".format(comm_thd, comm_time, overlapped_comp_time))
    return comm_time, overlapped_comp_time


def overlap1(comm_size, comp_time, comm_thd, still_time): # MB, ms #_excerbate_b150
    # model setting
    # '''  communication time  '''
    # 150Gb - 4096 thread - 50 * 1024 / 8 = 19200 MB - 1s
    # \delta * size / (alpha)^(k) / thread num = comm time
    # => 0.1 * 19200 / (0.99) ** (4096 / 64) / 4096* 1000
    # '''  computation time  '''
    # increase: (thread / 256) * 70 * \omega = 7 ms
    # => 0.000390625 * 256 * 70 =
    # => 0.99 ** (thread / 256) * (thread / 256) * t_overlap / 70 * 7
    # if 1 : 1, 0.99 -> 0.95




    # replace
    return _wrapped_overlap(comm_size, comp_time, comm_thd, still_time)


    new_bandwidth = 150 * 1024 / 8

    max_thd_usage = min(8192, comm_thd)
    comm_time1 = 0.1 * comm_size / (0.99) ** (max_thd_usage / 64) / max_thd_usage* 1000 # 0.05 * comm_size / (0.95) ** (max_thd_usage / 64) / max_thd_usage * 1000.0
    # print("comm1: {}".format(comm_time1))
    comm_time2 = comm_size / new_bandwidth * 1000.0  # a simple estimation
    comm_time = max([comm_time1, comm_time2])
    overlapped_comp_time = comp_time + 0.99 ** (comm_thd / 256) * (comm_thd / 256) * comm_time / 70 * 7 #todo: not accurate enough  #0.000390625 * comm_thd * comm_time
    if comm_time > overlapped_comp_time:
        # 0.99 ** (comm_thd / 256) * (comm_thd / 256) / 10 * x + comp_time = x
        # => x = comp_time / (1 - 0.99 ** (comm_thd / 256) * (comm_thd / 256) / 10)
        assert  (1 - 0.99 ** (comm_thd / 256) * (comm_thd / 256) / 70 * 7) > 0
        overlapped_comp_time = comp_time /  (1 - 0.99 ** (comm_thd / 256) * (comm_thd / 256) / 70 * 7)

    if comp_time == 0: overlapped_comp_time = 0

    #print("thd:{}, comm:{} , comp:{} ".format(comm_thd, comm_time, overlapped_comp_time))
    return comm_time, overlapped_comp_time