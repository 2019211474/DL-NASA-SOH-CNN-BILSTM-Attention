import os
from random import seed
from sklearn.metrics import r2_score
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Dropout, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import Model
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 我用的是MAC电脑的字符，Windows电脑可以改成黑体
set_random_seed(11)
seed(7)
SINGLE_ATTENTION_VECTOR = False

#建立一个CNN模型
def cnn_model(dr1, dense1, look_back):
    inputs = Input(shape=(look_back, INPUT_DIMS)) #输入层
    x = Conv1D(filters=8, kernel_size=1, activation='relu')(inputs)  # , padding = 'same'  #卷积层
    x = Dropout(dr1)(x)         #droupout层
    x = Flatten()(x)                #数据一维化
    output = Dense(dense1)(x)           #输出层
    output = Dense(pre_year, activation='linear')(output)
    model = Model(inputs=[inputs], outputs=output)
    return model
"""
参数：look_back

look_back: 表示使用过去多少个时间步来预测未来的值，即时间窗口的大小。
变量：

x：原始的特征数据集。
y：原始的标签数据集，即待预测的目标变量。在这里，y是通过对标签数据进行归一化得到的。
x_MinMax：MinMaxScaler()函数生成的实例对象，用于对特征数据x进行归一化处理。
y_MinMax：MinMaxScaler()函数生成的实例对象，用于对标签数据y进行归一化处理。
TrainX：训练特征数据集，由create_dataset()函数生成，shape为（样本数, 时间窗口大小, 特征数）。
Train_Y：训练标签数据集，由create_dataset()函数生成，shape为（样本数, 1），其中1表示目标变量y。
src：源数据文件所在路径。
train_path：源数据文件名。
df：DataFrame类型的原始数据集对象。
train_size：划分训练集和测试集时，训练集占比。
train：训练集的数据，包括特征和标签。
test：测试集的数据，包括特征和标签。
train_y：训练集的标签数据，不包括特征。
test_y：测试集的标签数据，不包括特征。
train_x：通过MinMaxScaler()函数对训练集的特征数据进行归一化得到的训练集特征数据。
test_x：通过MinMaxScaler()函数对测试集的特征数据进行归一化得到的测试集特征数据。
"""
# 定义数据归一化函数
def fit_size(x, y):
    from sklearn import preprocessing
    x_MinMax = preprocessing.MinMaxScaler()
    y_MinMax = preprocessing.MinMaxScaler()
    x = x_MinMax.fit_transform(x)
    y = y_MinMax.fit_transform(y)
    return x, y, y_MinMax

# 定义平铺函数
def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)


#数据处理转化成CNN可处理的形式
def create_dataset(dataset, train_y, look_back):  #train_y为标签即循环次数 TrainX训练特征则容量
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        y = train_y[(i + look_back):(i + look_back + 1), 0].tolist()
        dataY.append(y)
    TrainX = np.array(dataX)
    Train_Y = pd.DataFrame(dataY).values
    return TrainX, Train_Y

# 数据提取
src = r'/Users/xiongzhipeng/Downloads/毕业论文'
os.chdir(src)
train_path = 'B005放电数据集.csv'
train_path_oringe = 'B0005.csv'
# train_path = 'B0005.csv'
df = pd.read_csv(train_path, encoding='gbk')  # 自动把第一列作为标签列


# 数据切分
train_size = int(df.shape[0] * 0.7)
train = df.iloc[:train_size, :]
test = df.iloc[train_size:, :]
train_y = df.values[:train_size, 0].reshape(-1, 1) # 获取标签数据
test_y = df.values[train_size:, 0].reshape(-1, 1) # 获取标签数据

# 归一化
"""
这行代码调用了fit_size函数，将test和test_y作为参数传递
给该函数。fit_size函数将test和test_y进行归一化处理，并
返回归一化后的特征test_x、标签test_y以及标签的最大最
小值归一化器test_y_MinMax。因此，这行代码的作用是将
测试集数据test和test_y进行归一化处理，并将处理后的数
据和最大最小值归一化器存储在test_x、test_y和test_y_MinMax中。
"""
train_x, train_y, train_y_MinMax = fit_size(train, train_y)
test_x, test_y, test_y_MinMax = fit_size(test, test_y)

# TRAIN
INPUT_DIMS = train.shape[1]
# 预测未来1次的容量
pre_year = 1

# 调参方法
"""
使用optuna库的trial对象，定义了需要搜索的6个超参数，
分别是学习率lr、批次大小batch_size、第一层全连接层
神经元个数dense1、第一层Dropout层的比例dr1、训练
轮数epochs和滑动窗口大小look_back
"""
def objective(trial):
    lr = trial.suggest_uniform('lr', 0.002, 0.1) #学习率
    batch_size = trial.suggest_int('batch_size', 4, 6)#批个数
    dense1 = trial.suggest_int('dense1', 54, 99)
    dr1 = trial.suggest_uniform('dr1', 0.4, 0.6)#随机丢失率
    epochs = trial.suggest_int('epochs', 10, 40)#循环次数
    look_back = trial.suggest_int('look_back', 1, 2)
    train_X, train_Y = create_dataset(train_x, train_y, look_back)  #方法从训练集中生成时间序列数据，train_x是输入特征，train_y是目标值，look_back是时间步。
    train_Y_data = pd.DataFrame(train_Y)  #转化格式
    train_Y_data.dropna(inplace=True)  #删除目标值中的缺失值
    train_X = train_X[:train_Y_data.shape[0]]  #删除多余的特征值
    train_Y = train_Y_data.values #将DataFrame格式的目标值转换为NumPy数组。
    train_X = train_X.astype('float64')
    m = cnn_model(dr1, dense1, look_back)
    optimizer = Adam(lr)            # ADAM优化器
    m.compile(loss='mae', optimizer=optimizer)
    hist = m.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=0) #训练模型，使用训练集的数据进行训练，epochs为循环次数，batch_size为批大小，verbose=0表示不输出训练过程中的信息。
    mae = hist.history['loss'][-1] #指定最后一个损失值
    #rmae = mae / np.mean(train_y)
    return mae


""" 
把这一段反注释掉就可以开启调参模型，command + /
"""
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
# print('Best trial number: ', study.best_trial.number)
# print('Best value:', study.best_trial.value)  # 0.0476987
# print('Best parameters: \n', study.best_trial.params)
# parameters = study.best_trial.params  #将表现最佳的一次尝试的超参数的取值赋值给parameters变量，用于后续的模型训练。

# 找到的最好的参数
parameters =  {'lr': 0.002329824328245974, 'batch_size': 4, 'dense1': 85, 'dr1': 0.5777687768322071, 'epochs': 31, 'look_back': 2}

lr = parameters['lr']
dr1 = parameters['dr1']
look_back = parameters['look_back']
batch_size = parameters['batch_size']
dense1 = parameters['dense1']
epochs = parameters['epochs']

train_X, train_Y = create_dataset(train_x, train_y, look_back)
m = cnn_model(dr1, dense1, look_back)
optimizer = Adam(lr)
m.compile(loss='mae', optimizer=optimizer)
hist = m.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size)

test_X, test_Y = create_dataset(test_x, test_y, look_back)
test_X = test_X.astype('float64')
pred_y = m.predict(test_X)
mae = mean_absolute_error(test_Y, pred_y)
rmse = mean_squared_error(test_Y, pred_y)** 0.5
print("MAE = {} ".format(mae))
print("RMSE = {}".format(rmse))

# 反归一化并计算SOH
inv_pred_y = test_y_MinMax.inverse_transform(pred_y.reshape(-1, 1))
print(inv_pred_y)

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
plt.title('B0005_CNN_SOH预测')
plt.savefig("SOH_CNN_预测.png", dpi=500, bbox_inches='tight')
# 反注释后，画出使用Adam后的运行结果，需要把原来的绘图代注释掉
'''
plt.title('B0005_CNN_Adam_SOH预测')
plt.savefig("B0005_CNN_Adam_SOH预测.png", dpi=500, bbox_inches='tight')
'''
plt.show()
