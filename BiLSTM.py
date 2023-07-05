import os
from random import seed

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import CuDNNLSTM, Bidirectional

from tensorflow.python.keras.layers import LSTM, Bidirectional
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import Model

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
set_random_seed(11)
seed(7)
SINGLE_ATTENTION_VECTOR = False


def model_lstm(lstm_units, dr2, dense1, look_back):
    inputs = Input(shape=(look_back, INPUT_DIMS))

    # lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, activation='relu'), name='bilstm')(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    lstm_out = Dropout(dr2)(lstm_out)
    attention_mul = Flatten()(lstm_out)
    output = Dense(dense1)(attention_mul)
    output = Dense(pre_year, activation='linear')(output)
    model = Model(inputs=[inputs], outputs=output)
    return model


def fit_size(x, y):
    from sklearn import preprocessing
    x_MinMax = preprocessing.MinMaxScaler()
    y_MinMax = preprocessing.MinMaxScaler()
    x = x_MinMax.fit_transform(x)
    y = y_MinMax.fit_transform(y)
    return x, y, y_MinMax


# 预测未来1次的电池容量
def create_dataset(dataset, train_y, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        y = train_y[(i + look_back):(i + look_back + 1), 0].tolist()
        dataY.append(y)
    TrainX = np.array(dataX)
    Train_Y = pd.DataFrame(dataY).values
    return TrainX, Train_Y


src = r'/Users/xiongzhipeng/Downloads/毕业论文'
os.chdir(src)
train_path = 'B0005..csv'
df = pd.read_csv(train_path, encoding='gbk')
train_size = int(df.shape[0] * 0.7)
train = df.iloc[:train_size, :]
test = df.iloc[train_size:, :]
train_y = df.values[:train_size, 0].reshape(-1, 1)
test_y = df.values[train_size:, 0].reshape(-1, 1)

# 归一化
train_x, train_y, train_y_MinMax = fit_size(train, train_y)
test_x, test_y, test_y_MinMax = fit_size(test, test_y)

# TRAIN
INPUT_DIMS = train.shape[1]
# 预测未来1次的容量
pre_year = 1


# 调参方法
def objective(trial):
    lr = trial.suggest_uniform('lr', 0.002, 0.1)
    batch_size = trial.suggest_int('batch_size', 4, 6)
    lstm_units = trial.suggest_int('lstm_units', 33, 55)
    dense1 = trial.suggest_int('dense1', 54, 99)
    dr2 = trial.suggest_uniform('dr2', 0.2, 0.3)
    epochs = trial.suggest_int('epochs', 10, 40)
    look_back = trial.suggest_int('look_back', 1, 2)
    train_X, train_Y = create_dataset(train_x, train_y, look_back)
    train_Y_data = pd.DataFrame(train_Y)
    train_Y_data.dropna(inplace=True)
    train_X = train_X[:train_Y_data.shape[0]]
    train_Y = train_Y_data.values
    train_X = train_X.astype('float64')
    m = model_lstm(lstm_units, dr2, dense1, look_back)
    optimizer = Adam(lr)
    m.compile(loss='mae', optimizer=optimizer)
    hist = m.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=0)
    mae = hist.history['loss'][-1]
    return mae


""" 
把这一段反注释掉就可以开启调参模型，command + /
"""
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
# print('Best trial number: ', study.best_trial.number)
# print('Best value:', study.best_trial.value)
# print('Best parameters: \n', study.best_trial.params)
# parameters = study.best_trial.params
# 当前最优参数
# parameters = {'lr': 0.087376114153145, 'batch_size': 4, 'lstm_units': 37, 'dense1': 63, 'dr2': 0.24006829418417777, 'epochs': 34, 'look_back': 2} # 0.00467160
# parameters = {'lr': 0.006527407443894486, 'batch_size': 4, 'lstm_units': 55, 'dense1': 75, 'dr2': 0.23877285127597847, 'epochs': 39, 'look_back': 2} 0.13
parameters =  {'lr': 0.0065431526148670965, 'batch_size': 6, 'lstm_units': 42, 'dense1': 60, 'dr2': 0.22978364857534497, 'epochs': 39, 'look_back': 2}  # rmse = 0.004648060067110878

lstm_units = parameters['lstm_units']
lr = parameters['lr']
dr2 = parameters['dr2']
look_back = parameters['look_back']
batch_size = parameters['batch_size']
dense1 = parameters['dense1']
epochs = parameters['epochs']

train_X, train_Y = create_dataset(train_x, train_y, look_back)
m = model_lstm(lstm_units, dr2, dense1, look_back)
optimizer = Adam(lr)
m.compile(loss='mae', optimizer=optimizer)
hist = m.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size)

test_X, test_Y = create_dataset(test_x, test_y, look_back)
test_X = test_X.astype('float64')
pred_y = m.predict(test_X)
rmse = mean_squared_error(test_Y, pred_y)**0.5
mae = mean_absolute_error(test_Y, pred_y)
print("MAE = {} ".format(mae))
print("RMSE = {}".format(rmse))

inv_pred_y = test_y_MinMax.inverse_transform(pred_y.reshape(-1, 1))

# 下面内容做到容量计算SOH，使用原始数据——不加入可监测参数，使用可监测参数数据由于有SOH属性，故不需要下面的操作。
inv_pred_y = inv_pred_y / 2
df.iloc[:, 0] = df.iloc[:, 0] / 2       #df.iloc[:,0]表示df对象的所有第一列数据

# 画预测和真实值的对比图
plt.plot([x for x in range(train_size, 164)], inv_pred_y, color="red", label="pred")   # 使用原始数据——不加入可监测参数
# plt.plot([x for x in range(train_size, 50282)], inv_pred_y, color="red", label="pred")
plt.plot([x for x in range(167)], df.values[:, 0], color="blue", label="pred")   # 使用原始数据——不加入可监测参数
# plt.plot([x for x in range(50285)], df.values[:, 0], color="blue", label="pred")

plt.ylabel('锂电池SOH')
plt.xlabel('放电状态监测次数')
plt.legend(['SOH预测值', 'SOH真实值'], loc='best')
# 绘图
plt.title('B0005_BiLSTM_SOH预测')
plt.savefig("SOH_BiLSTM_预测.png", dpi=500, bbox_inches='tight')
# 反注释后，画出使用Adam后的运行结果，需要把原来的绘图代注释掉
'''
plt.title('B0005_BiLSTM_Adam_SOH预测')
plt.savefig("B0005_BiLSTM_Adam_SOH预测.png", dpi=500, bbox_inches='tight')
'''
plt.show()