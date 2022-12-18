from bayes_opt import BayesianOptimization
from perf_model import overlap_perf_comm_stage as perf_model
from perf_model import alloc_policy
from model import dl_model, cluster_model, dl_model_data
import numpy as np
import time


# global
model_to_opt = None
layer_num = -1
thread_allocation = []
best_iter_time = -1

def get_performance(**kwargs):
    global thread_allocation, best_iter_time
    tmp_allocation = []
    for li in range(layer_num):
        tmp_allocation.append(cluster_model.THREAD_OPTIONS[int(kwargs["l{}".format(li)])])
    plc = alloc_policy(tmp_allocation)
    iter_time = perf_model(model_to_opt, plc)
    if iter_time < best_iter_time or best_iter_time < 0:
        best_iter_time = iter_time
        thread_allocation = tmp_allocation
    return -iter_time

def bayes_alloc(model: dl_model, init_num = 10, iter_num = 300):
    global layer_num, model_to_opt
    layer_num = model.layer_num
    model_to_opt = model
    # Bounded region of parameter space
    pbounds = {}
    #allocation variables
    for i in range(layer_num):
        pbounds["l{}".format(i)] = (0, len(cluster_model.THREAD_OPTIONS) - 0.0001)


    optimizer = BayesianOptimization(
        f=get_performance,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=init_num,
        n_iter=iter_num,
    )

    print("Bayes opt time:", best_iter_time)
    print("Bayes opt policy:", thread_allocation)

    return thread_allocation, best_iter_time