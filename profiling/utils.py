'''some common functions'''

import psutil
import os
import math
import random
import torch.nn as nn

def get_mem_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # in MB


def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
      sx += x[i]
      sy += y[i]
      sxx += x[i]*x[i]
      syy += y[i]*y[i]
      sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return b,a,r


def get_random_seq(n):
    seq = ''
    for i in range(n): seq += str(random.randrange(10))
    return seq

def _model_to_list(model, modules):
    if len(list(model.children())) == 0:
        modules.append(model)
        return
    for module in model.children():
        _model_to_list(module, modules)

def model_to_list(model):
    modules = []
    _model_to_list(model, modules)
    model = nn.Sequential(*modules)
    return model

