from profiling.profiler import Profiler
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import timeit
import numpy as np
from profiling.utils import model_to_list, get_mem_usage



def disp_seq_model(seq_model):
    print(seq_model)
    layer_num = 0
    total_size = 0
    for n, l in enumerate(seq_model):  # model.named_children():
        # print(type(n), ":", type(l))
        layer_size = 0
        for params in l.parameters():
            tmp_size = 1
            for s in params.data.numpy().shape:
                tmp_size *= s
            layer_size += tmp_size
        total_size += layer_size
        # print("layer{}:\nsize:{}MB\n{}:\n{}\n\n".format(layer_num, layer_size * 4 / 1024 /1024, n, l))
        print("layer{}:\nsize:{}MB\n{}\n\n".format(layer_num, layer_size * 4 / 1024 / 1024, l))
        # print("layer{}: size-{} {}:{}".format(layer_num, layer_size, n, l))
        layer_num += 1
    print("total size: {}MB".format(total_size * 4 / 1024 / 1024))

def disp_model_chiledren(model):
    layer_num = 0
    total_size = 0
    layers = []
    for n, l in model.named_children():
        # print(type(n), ":", type(l))
        layers.append(l)
        layer_size = 0
        for params in l.parameters():
            tmp_size = 1
            for s in params.data.numpy().shape:
                tmp_size *= s
            layer_size += tmp_size
        total_size += layer_size
        # print("layer{}:\nsize:{}MB\n{}:\n{}\n\n".format(layer_num, layer_size * 4 / 1024 /1024, n, l))
        print("layer{}:\nsize:{}MB\n{}\n\n".format(layer_num, layer_size * 4 / 1024 / 1024, l))
        # print("layer{}: size-{} {}:{}".format(layer_num, layer_size, n, l))
        layer_num += 1
    print("total size: {}MB".format(total_size * 4 / 1024 / 1024))
    return layers


def profile(model):
    profiler = Profiler()
    print("Starting mem usage: {}MB".format(get_mem_usage()))
    model_info = profiler.profile(model)
    del model
    activ_size = 0
    layer_size = 0
    output_size = 0
    comp_time = 0
    bp_time = 0
    grad_size = 0
    for lid in range(len(model_info.keys())):
        print(
            "layer{:2}: size{:>10.6f}MB  activation:{:>10.6f}MB output_size:{:>10.6f}MB computation_time:{}ms    bp_time:{}ms grad_size:{:>10.6f}MB".format(
                lid,
                model_info[lid][0], model_info[lid][1], model_info[lid][2], model_info[lid][3], model_info[lid][4],
                model_info[lid][5]))
        layer_size += model_info[lid][0]
        activ_size += model_info[lid][1]
        output_size += model_info[lid][2]
        comp_time += model_info[lid][3]#[-1]
        bp_time += model_info[lid][4]#[-1]
        grad_size += model_info[lid][5]
    print(
        "Total size:{:>10.6f}MB  activation:{:>10.6f}MB   output_size:{:>10.6f}MB computation_time:{:>10.6f}ms  bp_time:{:>10.6f}ms grad_size:{:>10.6f}MB".format(
            layer_size, activ_size,
            output_size, comp_time, bp_time, grad_size))
    print("Ending mem usage: {:>10.6f}MB".format(get_mem_usage()))

if __name__ == "__main__":
    # Set up standard model.

    model = getattr(models, "vgg16")()
    layers = disp_model_chiledren(model)
    #data = torch.randn(4, 3, 224, 224)
    #model(data)
    profile(model)
    #seq_model = model_to_list(model)
    #profile(seq_model)
    '''
    vgg16 = models.vgg16(pretrained=False)
    vgg16_seq = torch.nn.Sequential(*(
            list(list(vgg16.children())[0]) +
            [torch.nn.AdaptiveAvgPool2d((7, 7)), torch.nn.Flatten()] +
            list(list(vgg16.children())[2])))
    profile(vgg16_seq)
    '''


