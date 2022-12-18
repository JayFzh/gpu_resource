import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD


# 生成随机点
def Produce_Random_Data():
    global x_data, y_data
    # 生成x坐标
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    #                                       增加一个维度
    # 生成噪声
    noise = np.random.normal(0, 0.02, x_data.shape)
    #                       均值 方差
    # 计算y坐标
    y_data = np.square(x_data) + noise
    # 画散点图
    plt.scatter(x_data, y_data)

#


model_path = "./profiling/model/nn_model"


def dnn(layer_num, unit_num, x_data, y_label, lr, training_iter = 300):
    model = tf.keras.Sequential()
    input_dim = x_data.shape[-1]
    model.add(tf.keras.layers.Dense(units=unit_num[0], input_dim=input_dim, activation='tanh'))
    for i in range(1, layer_num):
        model.add(tf.keras.layers.Dense(units=unit_num[i], activation='tanh'))
    model.compile(optimizer=SGD(lr), loss='mse')

    for i in range(training_iter):
        if i % 100 == 0: print("{} round finished.".format(i))
        loss = model.train_on_batch(x_data, y_label)
    y_pred = model.predict(x_data)
    #plt.plot(x_data, y_pred, 'r-', lw=5)
    model.save_weights(model_path)




# 1、生成随机点
Produce_Random_Data()
plt.show()
exit()
# 2、神经网络训练与预测
Neural_Network()

