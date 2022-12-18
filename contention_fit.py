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
raw_data_path = "../scripts/fitting_data/overlap2-processed"
x_norm_dim = [1000, 500, 10000]
y_norm_dim = 500
test_ratio = 0.3
# DNN model
#global model_path, comm_model, comp_model
model_path = None #"../profiling/model/comp_model"
comm_model = "../profiling/model/comm_model3"
comp_model = "../profiling/model/comp_model3"
layer_num = 3
layer_units = [5, 5, 1]

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


def dnn(layer_num, unit_num, x_data, y_label, lr = 0.1, training_iter = 100000):
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
    model.add(tf.keras.layers.Dense(units=output_dim))
    model.compile(optimizer=SGD(lr), loss='mse')

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
    plt.plot(losses[1000:])
    #print("Losses:", losses)
    #plt.plot(x_data, y_pred, 'r-', lw=5)
    model.save(model_path)
    plt.show()
    plt.plot(test_losses[10:])
    plt.show()

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
