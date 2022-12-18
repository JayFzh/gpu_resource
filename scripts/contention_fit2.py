import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD


'''
raw_data_path = "./profiling/data/organized"
x_norm_dim = [1000, 20, 10000]
y_norm_dim = 200
'''
raw_data_path = "../scripts/fitting_data/overlap-vgg16-comm"#processed"
x_norm_dim = [1000, 500, 10000]
y_norm_dim = 500
test_ratio = 0.3
# DNN model
#global model_path, comm_model, comp_model
model_path = None #"../profiling/model/comp_model"
comm_model = "../profiling/model/comm_model_nooverlap"#_mse_sigmoid2"#_mse_sigmoid"
comp_model = "../profiling/model/comp_model_nooverlap"

#comm_model = "../profiling/model/comm_model_small"#_mse_sigmoid2"#_mse_sigmoid"
#comp_model = "../profiling/model/comp_model_small"



# manual setting
layer_num = 3
layer_units = [5, 5, 1]

# bayes setting (comm - tanh -mse)
layer_num = 6
layer_units = [4, 8, 8, 8, 2, 1]

# bayes setting (comp - tanh -mse)
'''
unit_num:[4, 8, 4, 1]
lr:0.06576698175919443
iter_num:100000
activation:tanh
'''
layer_num = 4
layer_units = [4, 8, 4, 1]
#layer_num = 2
#layer_units = [4, 1]

# bayes setting (comp - sigmoid -mse)
'''
unit_num:[8, 8, 1]
lr:0.1
iter_num:100000
activation:sigmoid
'''

'''
Data loading
'''
def norm_data(data):
    new_data = data
    new_data[:, 0] = new_data[:, 0] / y_norm_dim
    new_data[:, 1] = new_data[:, 1] / y_norm_dim
    for i in range(len(x_norm_dim)):
        new_data[:, 2+i] = new_data[:, 2+i] / x_norm_dim[i]
    return new_data
def split_data(data):
    row_indices = np.random.permutation(data.shape[0])
    test_num = int((row_indices.shape[0])*test_ratio)
    x_test = data[row_indices[0:test_num], :]
    x_train = data[row_indices[test_num:], :]
    return x_train, x_test
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
    if normalize: data = norm_data(data)
    return split_data(data)

if __name__ == "__main__":
    train_data, test_data = load_data(raw_data_path)
    x_train = train_data[:,2:]
    x_test = test_data[:,2:]
    comm_train = train_data[:, 0]
    comm_test = test_data[:, 0]
    comp_train = train_data[:, 1]
    comp_test = test_data[:, 1]
'''
print(x)
print(comp_time)
print(comm_time)
exit()
'''

'''
Different fitting methods
'''
# linear regression
from scipy.optimize import curve_fit
def fun(X,coef_a,coef_b,coef_c,intercept):
    a=X[:,0]
    b=X[:,1]
    c=X[:,2]
    return coef_a*a+coef_b*b+coef_c*c+intercept

def fit_linear(x, y_label):
    x = x[0]
    y_label = y_label[0]
    res=curve_fit(fun, x, y_label,method="lm")
    print(res)
    return res

def res_func_comp(X):
    coef = [ 3.66896924e+02,  2.86386996e+00,  8.92102593e-03, -1.43781507e+05]
    coef = [-214.50223587,    0.26524523,    0.37025606,   84.33193517]
    coef_a = coef[0]
    coef_b = coef[1]
    coef_c = coef[2]
    intercept = coef[3]
    a = X[:, 0]
    b = X[:, 1]
    c = X[:, 2]
    return coef_a * a + coef_b * b + coef_c * c + intercept

def res_func_comm(X):
    coef = [-3.65734211e+02,  8.48154225e-01, -1.21365436e-02,  1.43455291e+05]
    coef = [-6.41415040e+02,  8.06228061e-02, -6.79515093e-01,  2.51894965e+02]
    coef_a = coef[0]
    coef_b = coef[1]
    coef_c = coef[2]
    intercept = coef[3]
    a = X[:, 0]
    b = X[:, 1]
    c = X[:, 2]
    return coef_a * a + coef_b * b + coef_c * c + intercept



def dnn(layer_num, unit_num, x_data, y_label, lr = 0.1, training_iter = 400000):
    test_x = x_data[1]
    test_y = y_label[1]
    x_data = x_data[0]
    y_label = y_label[0]
    model = tf.keras.Sequential()
    input_dim = x_data.shape[-1]
    output_dim = 1
    model.add(tf.keras.layers.Dense(units=unit_num[0], input_dim=input_dim, activation='tanh'))
    for i in range(1, layer_num-1):
        model.add(tf.keras.layers.Dense(units=unit_num[i], activation='tanh'))
    model.add(tf.keras.layers.Dense(units=output_dim, activation='sigmoid'))
    model.compile(optimizer=SGD(lr), loss='mae')

    losses = []
    test_losses = []
    for i in range(training_iter):
        if i % 100 == 0: print("{} round finished.".format(i))
        loss = model.train_on_batch(x_data, y_label)
        losses.append(loss)
        if i % 100 == 0:
            test_loss = model.test_on_batch(test_x, test_y)
            test_losses.append(test_loss)
    y_pred = model.predict(x_data)
    test_y_pred = model.predict(test_x)
    model.evaluate(x_data, y_label)
    model.evaluate(test_x, test_y)
    print("Train Prediction:", y_pred.reshape(len(y_pred)))
    print("Train Label:", y_label)
    print("Test Prediction:", test_y_pred.reshape(len(test_y_pred)))
    print("Test Label:", test_y)
    plt.plot(losses[40000:])
    #plt.plot(losses[100:])
    #print("Losses:", losses)
    #plt.plot(x_data, y_pred, 'r-', lw=5)
    model.save(model_path)
    plt.show()
    plt.plot(test_losses[100:])
    plt.show()


# Bayes opt range
# layer num 2 - 6
# unit_num = 4, 8, 16, 32
# lr = 0.01 - 0.1
# iteration = [10000, 300000]
# activation = ["tanh", "sigmoid"]
# https://blog.csdn.net/chenhepg/article/details/115721826 the choice of error functions - try "mae" or "mape"

'''
The besy policy found:
1. mape
unit_num:[2, 2, 1]
lr:0.001
iter_num:10000
activation:tanh

2. mae
unit_num:[8, 4, 16, 16, 2, 1]
lr:0.05446622805351858
iter_num:62628
activation:tanh

3.mse
unit_num:[4, 8, 8, 8, 2, 1]
lr:0.060753686948181285
iter_num:73886
activation:tanh
'''

#记录： comp_model训练
# lr: 0.00001
# iter: 100000
# loss: mape
def dnn_bo(layer_num, unit_num, x_data, y_label, lr = 0.1, training_iter = 10000, activation = "tanh", loss_function = "mse"):
    test_x = x_data[1]
    test_y = y_label[1]
    x_data = x_data[0]
    y_label = y_label[0]
    model = tf.keras.Sequential()
    input_dim = x_data.shape[-1]
    output_dim = 1
    model.add(tf.keras.layers.Dense(units=unit_num[0], input_dim=input_dim, activation=activation))
    for i in range(1, layer_num-1):
        model.add(tf.keras.layers.Dense(units=unit_num[i], activation=activation))
    model.add(tf.keras.layers.Dense(units=output_dim))
    model.compile(optimizer=SGD(lr), loss=loss_function)

    losses = []
    test_losses = []
    for i in range(training_iter):
        if i % 10000 == 0: print("{} round finished.".format(i))
        loss = model.train_on_batch(x_data, y_label)
        losses.append(loss)
        if i % 100 == 0:
            test_loss = model.test_on_batch(test_x, test_y)
            test_losses.append(test_loss)
    return test_losses[-1]

# optimize function structure with bayes optimization
from bayes_opt import BayesianOptimization
import time

if __name__ == "__main__":
    model_type = 1 # 0: comm 1: comp
    bo_x_train = x_train
    bo_x_test = x_test
    bo_y_train = [comm_train, comp_train][model_type]
    bo_y_test = [comm_test, comp_test][model_type]

def _get_near_integer(v):
    return int(v) #int(np.round(v))

def print_plc(policy):
    print("\n--------------------------")
    print("unit_num:{}".format(policy[1]))
    print("lr:{}".format(policy[2]))
    print("iter_num:{}".format(policy[3]))
    print("activation:{}".format(policy[4]))
    print("--------------------------\n")

def get_training_performance(**kwargs):
    unit_choice = [2, 4, 8, 16]
    activation_choice = ["tanh", "sigmoid"]
    iter_num_choice = 100000
    # get parameters
    layer_num = _get_near_integer(kwargs["layer_num"])
    unit_nums = [0 for i in range(layer_num)]
    unit_nums[-1] = 1
    for i in range(layer_num - 1):
        unit_nums[i] = unit_choice[_get_near_integer(kwargs["unit_num{}".format(i)])]
    lr = kwargs["lr"]
    iter_num = int(kwargs["iter"] * iter_num_choice)
    activation = activation_choice[_get_near_integer(kwargs["activation"])]

    plc = [layer_num, unit_nums, lr, iter_num, activation]
    print_plc(plc)
    loss = dnn_bo(layer_num, unit_nums, [bo_x_train, bo_x_test], [bo_y_train, bo_y_test], lr, iter_num, activation = activation)
    print("Get loss:{}".format(loss))
    global opt_loss, best_policy
    if loss < opt_loss:
        print("Better policy found.")
        best_policy = [layer_num, unit_nums, lr, iter_num, activation, loss]
        opt_loss = loss
    return loss * (-10000)

def bayes_opt():
    global opt_loss, best_policy
    opt_loss = np.inf
    best_policy = []
    # BO setting
    init_num = 5
    iter_num = 100
    # parameter range
    layer_num_range = (2, 7 - 1e-6)
    unit_num_range = (0, 4 - 1e-6)
    lr_range = (0.001, 0.1)
    iter_range = (0.1, 1)
    activation_range = (0, 2 - 1e-6)

    max_layer_num = 6
    pbounds = {}
    for li in range(max_layer_num):
        pbounds["unit_num{}".format(li)] = unit_num_range
    pbounds["layer_num"] = layer_num_range
    pbounds["lr"] = lr_range
    pbounds["iter"] = iter_range
    pbounds["activation"] = activation_range

    optimizer = BayesianOptimization(
        f=get_training_performance,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=init_num,
        n_iter=iter_num,
    )

def res_func_dnn(X):
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X)
    return y_pred


def func_dnn(comm_size, comp_size, thread_num, type = "comm"):
    if type == "comp" and comp_size == 0 : return 0
    if type == "comm": model_path = comm_model#"../profiling/model/comm_model2"
    elif type == "comp": model_path = comp_model#"../profiling/model/comp_model2"
    else: raise Exception("Wrong prediction type.")
    model = tf.keras.models.load_model(model_path)
    data = np.array([[0, 0, comm_size, comp_size, thread_num]])
    data = norm_data(data)
    data = data[:, 2:]
    y_pred = model.predict(data)
    #print(y_pred)
    #exit()
    print("overlap-{}: {}MB {}ms {} threads => {}ms".format(type, comm_size, comp_size, thread_num, y_pred[0][0] * y_norm_dim))
    return y_pred[0][0] * y_norm_dim

def func_dnn_group(inputs, type = "comm"):
    if type == "comm": model_path = comm_model
    elif type == "comp": model_path = comp_model
    else: raise Exception("Wrong prediction type.")
    model = tf.keras.models.load_model(model_path)
    data = np.array(inputs)
    data = np.insert(data, 0, 0, axis = 1)
    data = np.insert(data, 0, 0, axis = 1)
    #print(data)
    data = norm_data(data)
    data = data[:, 2:]
    y_pred = model.predict(data)
    #print(y_pred)
    #exit()

    res = []
    for ri, r in enumerate(y_pred):
        if type == "comp" and inputs[ri][1] == 0: res.append(0)
        else: res.append(y_pred[ri][0] * y_norm_dim)
    return res


from scripts.run_data_analyze import comm_resource, comp_resource_curve, comm_resource_opt, comm_resource_opt2
from model import cluster_model
def _fitting_comm(x):
    # [0.01804497 6.09254452 1. ]
    # [for getting curve]
    bw_ratio = cluster_model.bw_incre
    x[-1] = x[-1] / bw_ratio

    # 1 /x
    popt = [0.01804497, 6.09254452, 1. ]
    func = comm_resource
    # 1 / (a + bn) ** c
    popt = [ 3.3410026, -202.96018144, 0.22023362, 0.]#[ 1.31388620e+01, -1.55065897e+03,  1.81770984e-01,  0.00000000e+00]
    #func = comm_resource_opt
    #popt = [1.13759935e+01, -1.33651546e+03, 1.87308009e-01, 2.07182664e+00]
    # fixed
    popt = [  0.56570925, -33.01543378, 0.27716796, 0.        ]
    popt = [8.2216991, 6.53459963, 0.0290902, 0.5]  # 4 = 0.6
    # to delete - debug
    #popt = [ 1.84282801e+04, -1.84346273e+04,  1.00117551e-04,  3.00000000e+00]

    #popt = [9.45899324, 7.89500341, 0.02730211, 0.5]  # 4 -xception

    #popt = [9.05115419, 7.65599097, 0.03109309, 0.5]  # 4 - bert-large

    #popt = [13.70164828, 8.82019619, 0.07261649, 0.5]  # 4 - resnet152

    func = comm_resource_opt2

    all_x = np.array([x])
    #print(all_x)
    #print(func(all_x, a=popt[0], b=popt[1], c=popt[2], d=popt[3]))

    return func(all_x, a=popt[0], b=popt[1], c=popt[2], d=popt[3])[0] / bw_ratio



def _fitting_comp(x):
    #[1.00000000e+00 9.11320012e-01 3.29246779e-04]
    popt = [1.00000000e+00,8.99146454e-01,1.76396838e-04, 0]#[1.00000000e+00, 9.11320012e-01, 3.29246779e-04]
    popt = [-3.47675831e+03, -2.81280320e+03, -5.12178748e-01 , 1.00000000e+00]
    #popt = [1.00000000e+00 , 8.70344381e-01, 1.41829103e-04, 0]
    # fixed
    popt = [1.00000000e+00,  8.76321235e-01,  1.60589190e-04, -2.04574372e-04]
    #popt = [ 1.00000000e+00, 8.99186533e-01, 1.77585629e-04, -1.75034667e-04]

    #popt = [ 1.27640285e+00,  1.28971792e+00,  5.83942990e-05, -2.04574372e-04]  # 4 - xception
    #popt = [7.42824887e-03, 6.11353445e-03, 1.06948925e-06,-2.04574372e-04]  # bert-large [ 1.00000000e+00  8.23011528e-01  1.43975958e-04 -2.75400536e-02]

    #popt = [1.00000000e+00, 7.67963019e-01, 1.25776191e-04, -4.26842529e-02]  # resnet-152
    #popt = [1.00000000e+00, 7.75813181e-01, 1.23086030e-04, -1.47185678e-02] # resnet-152 v2

    func = comp_resource_curve
    all_x = np.array([x])
    #print(all_x)
    res = func(all_x, a=popt[0], b=popt[1], c=popt[2], d=popt[3])[0]
    if res <0: res = np.inf
    return res

def _fitting_unoverlap_comm(x):
    # [0.04302182 6.73128312 1. ]
    # [for getting curve]
    bw_ratio = cluster_model.bw_incre
    x[-1] = x[-1] / bw_ratio

    # 1 / x
    popt = [0.04302182, 6.73128312, 1. ]
    func = comm_resource

    # 1 / (a + bn) ** c
    popt = [7.86459403e+03, -4.96852926e+05, 1.20545194e-01, 0]
    #func = comm_resource_opt
    # d + 1 / (a + bn) ** c
    #popt = [5.13131638e+03, -3.23642926e+05, 1.28124071e-01, 4.07446624e+00]
    popt = [10.01228313, 7.10248977, 0.05409201, 0.5]  # 4 = 0.6

    #popt = [11.2868949, 7.16382211, 0.08027396, 0.5]  # 4 - xception

    #popt = [14.90506948, 9.78315476, 0.0662079, 0.5]  # 4 - bert-large

    #popt = [13.81307014, 7.53180825, 0.07115125, 0.5]  # 4 - resnet152

    func = comm_resource_opt2

    all_x = np.array([x])
    return func(all_x, a=popt[0], b=popt[1], c=popt[2], d=popt[3])[0] / bw_ratio

def func_fitting(comm_size, comp_size, thread_num, type = None, const_time = 0):
    #if type == "comm": return func_dnn(comm_size, comp_size, thread_num, type = "comm")
    #comm_size = inputs[0]
    #comp_size = inputs[1]
    #thread_num = inputs[2]
    if comm_size == 0 and comp_size == 0: return 0,0
    full_overlap_comm_time = _fitting_comm([comm_size, thread_num])
    full_overlap_comp_time = _fitting_comp([comp_size, thread_num])
    nooverlap_comm_time = _fitting_unoverlap_comm([comm_size, thread_num])
    nooverlap_comp_time = comp_size
    print("{} {} {} {} {} thd:{}".format(full_overlap_comm_time + const_time, full_overlap_comm_time, full_overlap_comp_time, nooverlap_comm_time, nooverlap_comp_time, thread_num))
    #print("overlap: {}MB {}ms {} threads".format(comm_size, comp_size, thread_num))
    if full_overlap_comp_time < 0 or full_overlap_comm_time < 0 or nooverlap_comp_time <0 or nooverlap_comm_time < 0:
        exit()
    # see which is the early finish one
    if (full_overlap_comm_time + const_time) <= full_overlap_comp_time:
        comm_time = full_overlap_comm_time + const_time
        comp_time = comm_time + nooverlap_comp_time * ( 1 - comm_time / full_overlap_comp_time)
    else:
        comp_time = full_overlap_comp_time
        # lyz - lat
        remaining_const = max(0, const_time - comp_time)
        overlapped_comm_t = max(0, comp_time - const_time)
        comm_time = comp_time + remaining_const + nooverlap_comm_time * (1 - overlapped_comm_t / full_overlap_comm_time)
    if full_overlap_comp_time == 0 and full_overlap_comm_time == 0: exit()
    #print("overlap: {}MB {}ms {} threads => comm: {}ms comp: {}ms".format(comm_size, comp_size, thread_num, comm_time, comp_time))
    if not type: return comm_time, comp_time
    elif type == "comm": return comm_time
    elif type == "comp": return comp_time
    else: assert False

def func_fitting_group(inputs, type = "comm", const_time = 0):
    #if type == "comm": return func_dnn_group(inputs, type="comm")
    data = np.array(inputs)
    res_comm = []
    res_comp = []
    for entry in data:
        comm_t, comp_t = func_fitting(entry[0], entry[1], entry[2], const_time = const_time)
        res_comm.append(comm_t)
        res_comp.append(comp_t)
    if type == "comm":
        return res_comm
    elif type == "comp":
        return res_comp
    else:
        return res_comm, res_comp



'''
Validation
'''
def validate(func, x, y_label, mark):
    print("{} fit:".format(mark))
    res = func(x)
    res = res.reshape(len(res))
    diff = np.abs(res - y_label)
    diff_ratio = np.abs(res - y_label) / y_label * 100.0
    avg_error = np.mean(diff_ratio)
    print("Avg error: {}%".format(avg_error))
    print("Max error: {}%".format(max(diff_ratio)))
    print("Min error: {}%".format(min(diff_ratio)))

if __name__ == "__main__x":
    bayes_opt()

if __name__ == "__main__":
    mode = ["fit", "test"][0]
    method = ["lr", "dnn"][1]
    target = ["comp", "comm"][1]
    #x_data = [x_train, x_test]
    #y_data = [[comp_train, comp_test], [comm_train, comm_test]]
    #global model_path, comm_model, comp_model
    if target == "comp": model_path = comp_model
    else: model_path = comm_model

    if mode == "fit":
        x = [x_train, x_test]
        if target == "comp": label = [comp_train, comp_test]
        else: label = [comm_train, comm_test]
        if method == "lr":
            fit_linear(x, label)
        elif method == "dnn":
            dnn(layer_num, layer_units, x, label)
    elif mode == "test":
        if target == "comp": label = [comp_train, comp_test]
        else: label = [comm_train, comm_test]
        if method == "lr":
            validate(res_func_comp, x_train, comp_train, "Computation-train")
            validate(res_func_comm, x_train, comm_train, "Communication-train")
            validate(res_func_comp, x_test, comp_test, "Computation-test")
            validate(res_func_comm, x_test, comm_test, "Communication-test")
        elif method == "dnn":
            validate(res_func_dnn, x_train, label[0], "Train")
            validate(res_func_dnn, x_test, label[1], "Test")


'''
print("Computation fit:")
res = res_func_comp(x)
diff = np.abs(res - comp_time)
diff_ratio = np.abs(res - comp_time) / comp_time * 100.0
avg_error = np.mean(diff_ratio)
print("Avg error: {}%".format(avg_error))
print("Max error: {}%".format(max(diff_ratio)))
print("Min error: {}%".format(min(diff_ratio)))

print("Communication fit:")
res = res_func_comm(x)
diff = np.abs(res - comm_time)
diff_ratio = np.abs(res - comm_time) / comm_time * 100.0
avg_error = np.mean(diff_ratio)
print("Avg error: {}%".format(avg_error))
print("Max error: {}%".format(max(diff_ratio)))
print("Min error: {}%".format(min(diff_ratio)))
#'''


'''
# fitting methods
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


x = np.array(df.iloc[:,0:4].values)
y = np.array(df.iloc[:,5].values)


poly_reg =PolynomialFeatures(degree=2) #三次多项式
X_ploy =poly_reg.fit_transform(x)
lin_reg_2=linear_model.LinearRegression()
lin_reg_2.fit(X_ploy,y)
predict_y =  lin_reg_2.predict(X_ploy)
strError = stdError_func(predict_y, y)
R2_1 = R2_1_func(predict_y, y)
R2_2 = R2_2_func(predict_y, y)
score = lin_reg_2.score(X_ploy, y) ##sklearn中自带的模型评估，与R2_1逻辑相同

print("coefficients", lin_reg_2.coef_)
print("intercept", lin_reg_2.intercept_)
print('degree={}: strError={:.2f}, R2_1={:.2f},  R2_2={:.2f}, clf.score={:.2f}'.format(
    3, strError,R2_1,R2_2,score))
'''
