import numpy as np
from typing import List
import random

class cluster_model:
    #bandwidth = 150 * 1024 / 8 # MB/s
    bandwidth = 50 * 1024 / 8 # MB/s
    latency = 10 # ms per activation

    baseline_bandwidth = 50
    baseline_worker_n = 2

    # change the configs
    config_bandwidth = 50 # deprecated (replaced by curve_bandwidth)
    config_worker_n = 2
    base_lat =  0.005#1.4 / 15 #* 0.001# ms - the effect of different latency
    allreduce_algo = "ring" # use ringAllReduce for now

    #another bandwidth config
    curve_bandwidth = 50
    THREAD_OPTIONS = [64, 128, 256, 512, 1024, 2048, 4096]
    bw_incre = 1

    def __init__(self):
        pass

    @staticmethod
    def ring_allreduce_size(n):
        #[debugging]
        #n = 2
        return 2 * (n - 1) / n

    @staticmethod
    def cluster_comm_size(comm_size):
        base_size   = cluster_model.ring_allreduce_size(cluster_model.baseline_worker_n)
        config_size = cluster_model.ring_allreduce_size(cluster_model.config_worker_n)
        return config_size / base_size * comm_size * cluster_model.baseline_bandwidth / cluster_model.config_bandwidth

    @staticmethod
    def cluster_comm_lat():
        return (2 * cluster_model.config_worker_n - 2) * cluster_model.base_lat

class dl_model_data:
    def __init__(self):
        self.layer_size = None
        self.layer_computation_time = None

class dl_model:
    def __init__(self, model_data: dl_model_data = None, random_layer_num = 0,
                 random_size_range = None, random_comp_range = None, fusion_num = 0):
        self.layer_num = 0
        self.layer_size = [] # MB
        self.layer_computation_time = [] # ms
        # dp data
        self.discrete_comp_time = None
        if model_data is not None: self.load(model_data)
        elif random_layer_num <= 0 or random_size_range is None or random_comp_range is None:
            raise Exception("Must specify model info or random info!")
        else:
            self.load_random(random_layer_num, random_size_range, random_comp_range)
        #self.set_discrete_comp_time(discrete_num)
        # keep track of fusion group
        self.fusion_group = [1 for i in range(self.layer_num)]
        # fusion model layers if required
        if fusion_num > 0 and fusion_num < self.layer_num: self.fusion_model(fusion_num)

    # load given model info
    def load(self, model_data: dl_model_data):
        assert model_data.layer_computation_time != None and model_data.layer_size != None
        assert len(model_data.layer_size) == len(model_data.layer_computation_time)
        self.layer_num = len(model_data.layer_size)
        self.layer_size = model_data.layer_size
        self.layer_computation_time = model_data.layer_computation_time

    # randomly fuses the model
    def random_fusion_model(self, max_fusion_num):
        layer_remain = self.layer_num
        group_spec = []
        while layer_remain > 0 and len(group_spec) < (max_fusion_num - 1):
            fusion_size = random.randint(1, layer_remain)
            layer_remain -= fusion_size
            group_spec.append(fusion_size)
            #print("Layer remain:", layer_remain, " max fusion num:", max_fusion_num)
        if layer_remain > 0: group_spec.append(layer_remain)
        print("random group spec:{}".format(group_spec))
        self.group(group_spec)

    # fusion the model to designated number of groups
    def fusion_model(self, fusion_num):
        group_size = self.layer_num // fusion_num
        group_spec = [group_size for i in range(fusion_num)]
        remain_size = self.layer_num - fusion_num * group_size
        for i in range(remain_size):
            group_spec[i] += 1
        self.group(group_spec)
        '''
        group_size= int(np.ceil(self.layer_num / fusion_num))
        fused_size = [sum(self.layer_size[i * group_size : (i + 1) * group_size]) for i in range(fusion_num)]
        fused_time = [sum(self.layer_computation_time[i * group_size : (i + 1) * group_size]) for i in range(fusion_num)]
        self.layer_num = fusion_num
        self.layer_size = fused_size
        self.layer_computation_time = fused_time
        '''
    # fusion the model to designated groups
    def group(self, group_size):
        fused_size = []
        fused_time = []
        fused_group_size = []
        for gi in range(len(group_size)):
            #print("{} -> {}".format(sum(group_size[:gi]), sum(group_size[:gi + 1])))
            fused_size.append(sum(self.layer_size[sum(group_size[:gi]) : sum(group_size[:gi + 1])]))
            fused_time.append(sum(self.layer_computation_time[sum(group_size[:gi]): sum(group_size[:gi + 1])]))
            fused_group_size.append(sum(self.fusion_group[sum(group_size[:gi]) : sum(group_size[:gi + 1])]))
        #exit()
        self.layer_num = len(group_size)
        self.layer_size = fused_size
        self.layer_computation_time = fused_time
        self.fusion_group = fused_group_size
        #print("Grouped: {} {}".format(self.layer_computation_time, self.fusion_group))

    # generate random model data
    def load_random(self, layer_num, size_range: List, comp_range: List): #MB, ms
        self.layer_num = layer_num
        self.layer_computation_time = [(np.random.randint(1, 1000) / 1000.0 * (comp_range[1] - comp_range[0]) + comp_range[0]) for i in range(layer_num)]
        self.layer_size = [(np.random.randint(1, 1000) / 1000.0 * (size_range[1] - size_range[0]) + size_range[0]) for i in range(layer_num)]
        print("{} layer generated.".format(layer_num))
        print("Total size: {}MB".format(sum(self.layer_size)))
        print("Total computation time: {}ms".format(sum(self.layer_computation_time)))
        print(self.layer_size)
        print(self.layer_computation_time)
        exit()

    # '''
    #   dp related
    # ''''
    # discretize the total computation
    def set_discrete_comp_time(self, z):
        self.discrete_comp_time = [sum(self.layer_computation_time)  / z + 0.0001 for i in range(z)]

    def get_discrete_comp_time(self):
        assert self.discrete_comp_time is not None
        return self.discrete_comp_time

    # the minimum computation chunk index with time larger than the given layer group index
    # at which chunk the computation is finished
    def get_cell_comp_index(self, group_index):
        group_end_time = sum(self.layer_computation_time[:group_index + 1])
        #print("sum1:", group_end_time)
        #print("sum2:", sum(self.discrete_comp_time))
        return (np.cumsum(self.discrete_comp_time) >= group_end_time).tolist().index(1)
