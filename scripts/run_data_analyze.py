import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
Data loading
'''
def load_data(path, normalize = True):
    #[comm_time, comp_time, comm_size, comp_size, thread_alloc, bandwidth]
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            entry = []
            if len(line) > 1:
                raw_data = line.split()
                for d in raw_data[:-2]: entry.append(eval(d))
                data.append(entry)
    data = np.array(data)
    return data

'''
Curve fit
'''

def comm_resource_opt(x, a, b, c):
    comm_size = x[:, 0:1] # MB
    thread_alloc = x[:, 1:2] # num of threads
    #print(comm_size)
    #print(thread_alloc)
    #exit()
    comm_time = comm_size / (b + a * thread_alloc) ** c
    comm_time = comm_time.reshape(len(comm_time))
    #transfer_time = comm_size / b
    #res = np.array([[comm_time[i],transfer_time[i][0]] for i in range(len(comm_time))])
    #comm_time = np.max(res, axis=1)
    return comm_time

def comm_resource_opt2(x, a, b, c, d):
    comm_size = x[:, 0:1] # MB
    thread_alloc = x[:, 1:2] # num of threads
    #print(comm_size)
    #print(thread_alloc)
    #exit()
    comm_time = d + comm_size / (b + a * thread_alloc) ** c
    #comm_time = d + comm_size / (b + a * thread_alloc ** c)
    comm_time = 0 + comm_size / (b - a * np.exp(-c * thread_alloc ** (0.6)))  # 4
    comm_time = comm_time.reshape(len(comm_time))
    #transfer_time = comm_size / b
    #res = np.array([[comm_time[i],transfer_time[i][0]] for i in range(len(comm_time))])
    #comm_time = np.max(res, axis=1)
    return comm_time


def comm_resource(x, a, b, c, d):
    comm_size = x[:, 0:1] # MB
    thread_alloc = x[:, 1:2] # num of threads
    #print(comm_size)
    #print(thread_alloc)
    #exit()
    comm_time = 0 + comm_size / (b + a * thread_alloc) ** c
    comm_time = comm_time.reshape(len(comm_time))
    #transfer_time = comm_size / b
    #res = np.array([[comm_time[i],transfer_time[i][0]] for i in range(len(comm_time))])
    #comm_time = np.max(res, axis=1)
    return comm_time

def comm_unoverlap():
    pass

def comp_resource_linear(x, a, b, c):
    pass

def comp_resource_curve(x, a, b, c, d):
    comp_size = x[:, 0:1]  # ms
    thread_alloc = x[:, 1:2]  # num of threads
    comp_time = a* comp_size / (b - c * thread_alloc)
    comp_time = comp_time.reshape(len(comp_time))
    return comp_time

def comp_resource_curve2(x, a):
    comp_size = x[:, 0:1]  # ms
    thread_alloc = x[:, 1:2]  # num of threads
    comp_time = comp_size / (1 - a * thread_alloc)
    comp_time = comp_time.reshape(len(comp_time))
    return comp_time


if __name__ == "__main__(plot)":
    raw_data_path = "../scripts/fitting_data/overlap-vgg16-processed"
    var_names = ["Comm size(MB)", "Comp size(ms)", "Thread num"]
    var_dim_index = 4  #2, 3, 4 - comm_size, comp_size, thread_allocation
    all_dim_indexes = [2, 3, 4]
    fix_dim_indexes = list(all_dim_indexes)
    fix_dim_indexes.pop(fix_dim_indexes.index(var_dim_index))
    target_dimension = 0 #0, 1  - comm_time, comp_time
    data = load_data(raw_data_path)
    processed_data = {}
    for entry in data:
        key1 = np.round(entry[fix_dim_indexes[0]], 0)
        key2 = np.round(entry[fix_dim_indexes[1]], 0)
        if key1 not in processed_data.keys():
            processed_data[key1] = {}
        if key2 not in processed_data[key1].keys():
            processed_data[key1][key2] = [[], [], []] # var, comm_time, comp_time
        var = entry[var_dim_index]
        processed_data[key1][key2][0].append(var)
        processed_data[key1][key2][1].append(entry[0])
        processed_data[key1][key2][2].append(entry[1])

    # plot each curve - only those with num of points >= 3
    for k1 in processed_data.keys():
        for k2 in processed_data[k1].keys():
            vars = np.array(processed_data[k1][k2][0])
            print("fix:{}-{}    len:{}".format(k1, k2, len(vars)))
            if len(vars) < 3: continue
            target1 = np.array(processed_data[k1][k2][1])
            target2 = np.array(processed_data[k1][k2][2])
            sort_args = np.argsort(vars)
            vars = vars[sort_args]
            target1 = target1[sort_args]
            target2 = target2[sort_args]

            #plt.figure(figsize=(4.2, 2.8))
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            fix_names = list(var_names)
            fix_names.pop(all_dim_indexes.index(var_dim_index))
            ax1.set_title("Fixed: {} - {} : {} - {}".format(fix_names[0], k1, fix_names[1], k2))
            ax1.set_xlabel(var_names[all_dim_indexes.index(var_dim_index)])
            ax1.set_ylabel('comm time (ms)', color=color)
            ax1.plot(vars, target1, color=color, marker ="*")
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xticks(vars)
            ax1.set_xticklabels([int(v) for v in vars], rotation=60)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('comp time (ms)', color=color)
            ax2.plot(vars, target2, color=color, marker = "o")
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()
            plt.show()


def mape(pred, label):
    error = np.mean(np.abs(pred - label) / label)
    return error


# use DNN model prediction
from scripts.contention_fit2 import func_dnn as predict_func
from scripts.contention_fit2 import func_dnn_group as predict_func_group
def dnn_predict(x, type = "comp"):
    res = predict_func_group(x, type = type)
    return res

# unoverlap communication
if __name__ == "__main__": # (sets of params)
    raw_data_path = "../scripts/fitting_data/overlap-vgg16-processed"#unoverlap-comm-processed"#nooverlap"#
    data = load_data(raw_data_path)
    var_dim_index = 4
    target_dim_index = 1
    names = ["comm_size", "comp_size"]
    y_names = ["comm time (ms)", "comp size (ms)"]

    if target_dim_index == 0:
        func = comm_resource
    else:
        func = comp_resource_curve

    #all_dim_indexes = [2, 3, 4]
    processed_data = {}
    for entry in data:
        comm_time = entry[0]
        comp_time = entry[1]

        comm_size = np.round(entry[2], 1)
        comp_size = np.round(entry[3], 1)

        #66.71, 69.14, 71.96
        if target_dim_index == 0:
            target = comm_time
            other = comp_time
            target_key = comm_size
            other_key = comp_size
        else:
            target = comp_time
            other = comm_time
            target_key = comp_size
            other_key = comm_size
        if target > other and not other == 0:
            #continue
            print("new entry generated")
            if target_dim_index == 0:
                #continue
                from scripts.contention_fit2 import _fitting_unoverlap_comm
                remain_comm_time = target - other
                thd = entry[var_dim_index]
                original_comm_time = _fitting_unoverlap_comm([comm_size, thd])
                overlap_comm_time = int(original_comm_time - remain_comm_time)
                print("\noriginal:{} remain:{} overlap comm:{}".format(original_comm_time, remain_comm_time, overlap_comm_time))
                target_key = target_key * overlap_comm_time / original_comm_time
                target = other
            else:
                # or we can deduct the
                #exit()
                remain_comp_time = target - other
                overlap_comp_time = max(np.round(target_key - remain_comp_time, 1), 0)
                print("\noriginal:{} remain:{} overlap comm:{}".format(target_key, remain_comp_time,
                                                                       overlap_comp_time))
                target_key = overlap_comp_time
                target = other

        print(entry)
        if target_key == 0:
            print("\nzero, continue!!\n")
            continue
        if target_key not in processed_data.keys():
            processed_data[target_key] = [[],[], other_key]
        var = entry[var_dim_index]
        if var not in processed_data[target_key][0]:
            processed_data[target_key][0].append(var)
            processed_data[target_key][1].append(target)
        print("size:{} thd:{} time:{}".format(target_key, var, target))

    all_x = []
    all_y = []
    all_dnn_x = []
    sets_of_error = []
    dnn_error = []
    for k1 in processed_data.keys():
        filter = [14, 28, 54, 11, 19, 37, 67, 21, 39, 22, 69, 41]
        #if k1 not in filter: continue

        vars = np.array(processed_data[k1][0])
        print("fix:{}    len:{}".format(k1, len(vars)))
        #if len(vars) < 4: continue
        target1 = np.array(processed_data[k1][1])
        sort_args = np.argsort(vars)
        vars = vars[sort_args]
        target1 = target1[sort_args]

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        fix_name = names[target_dim_index]
        ax1.set_title("Fixed: {} - {}ms ".format(fix_name, k1))
        ax1.set_xlabel("Thread num")
        ax1.set_ylabel(y_names[target_dim_index], color=color)
        ax1.plot(vars, np.array(target1), color=color, marker="*")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(vars)
        ax1.set_xticklabels([int(v) for v in vars], rotation=60)

        # fit the curve
        x = []
        y = []
        dnn_x = []
        #dnn_y = []
        for vi, v in enumerate(vars):
            x.append([k1, v])
            y.append(target1[vi])
            all_x.append([k1, v])
            all_y.append(target1[vi])
            if target_dim_index == 1:
                dnn_x.append([processed_data[k1][2], k1, v])
                all_dnn_x.append([processed_data[k1][2], k1, v])
            else:
                dnn_x.append([k1, processed_data[k1][2], v])
                all_dnn_x.append([k1, processed_data[k1][2], v])
        continue

        x = np.array(x)
        y = np.array(y)
        #print(x)
        #print(y)
        #popt, pcov = curve_fit(func, x, y, p0=[2.23582881e-03, 0.0, 1.0, 0.0], maxfev=5000)
        popt = [ 3.3410026, -202.96018144, 0.22023362, 0.]# [ 7.86455647e+03, -4.96850540e+05, 1.20545233e-01, 0.00000000e+00]#[ 1.31388620e+01, -1.55065897e+03, 1.81770984e-01, 0.00000000e+00]# [ 5.13131638e+03, -3.23642926e+05, 1.28124071e-01, 4.07446624e+00]#[ 7.86459403e+03, -4.96852926e+05, 1.20545194e-01]#[-2.14750070e+04,-1.95706036e+04, -7.07057688e+00]
        pred = func(x, a = popt[0], b=popt[1], c = popt[2], d = popt[3])
        if target_dim_index == 1: dnn_pred = dnn_predict(dnn_x)
        else: dnn_pred = dnn_predict(dnn_x, type="comm")
        #print(popt/popt[0])
        print(popt)
        print(pred)
        print(dnn_pred)
        print(y)
        ax1.plot(vars, np.array(pred), marker = 'o', label = "fitted curve")
        ax1.plot(vars, np.array(dnn_pred), marker='^', label="dnn pred")

        sets_of_error.append(mape(pred, y))
        dnn_error.append(mape(dnn_pred, y))
        #exit()

        fig.tight_layout()
        plt.legend()
        plt.show()

    # 14, 28, 54, 11, 19, 37, 67, 21, 39, 22, 69, 41
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    #print(len(all_x))
    #exit()
    popt, pcov = curve_fit(func, all_x, all_y, maxfev=5000)#, p0=[2.23582881e-03, 0.0, 1.0, 0.0])#, p0=[1.73582881e-04])
    #popt =  [ 5.13131638e+03, -3.23642926e+05, 1.28124071e-01, 4.07446624e+00]#[ 7.86459403e+03, -4.96852926e+05, 1.20545194e-01]#[-2.14750070e+04, -1.95706036e+04, -7.07057688e+00]
    pred = func(all_x, a=popt[0], b=popt[1], c=popt[2], d = popt[3])
    all_error = mape(pred, all_y)

    if target_dim_index == 1:
        dnn_pred = dnn_predict(all_dnn_x)
    else:
        dnn_pred = dnn_predict(all_dnn_x, type="comm")
    dnn_error = mape(dnn_pred, all_y)

    print("All fit params:{} {}".format(popt/ popt[0], popt))
    print("All error:{}".format(all_error))
    print("Mean error:{}".format(np.mean(sets_of_error)))
    print("DNN error:{}".format(np.mean(dnn_error)))


