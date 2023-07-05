import os
from random import seed
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.layers import Dense, Conv1D
from tensorflow.python.keras.layers import Permute, Multiply,Reshape,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Bidirectional, LSTM
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import Model
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
set_random_seed(11)
seed(7)


def attention_3d_block(inputs):                         # inputs：输入张量，它的形状为(batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])                    # input_dim：输入张量中的特征数。
    a = inputs                                                      # a：将输入张量复制一份，用于计算注意力权重
    a = Dense(input_dim, activation='softmax')(a)       # Dense(input_dim, activation='softmax')：一个全连接层，它将输入张量的每个特征
                                                                                     # 映射到一个注意力权重值，使用softmax激活函数将权重归一化，以使它们的和等于1。
    a_probs = Permute((1, 2), name='attention_vec')(a)      # 使用Permute层将注意力权重的维度从(batch_size, time_steps, input_dim)转换为
                                                                                         # (batch_size, input_dim, time_steps)，以便在下一步中将其与输入张量相乘。
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

"""
一维卷积层可以捕获输入序列中的一些局部模式和特征，这些模式和特征可以帮助模型更好地理解序列数据。扁平化层可以将卷积层的输出转换为一维张量，这有助于后续的全连接层进行预测。
全连接层可以对输入的一维特征向量进行非线性变换，从而构造一个更复杂的模型。这些层可以通过学习不同的权重和偏置值来捕获序列中的不同模式和特征，并将它们组合在一起以产生最终的预测结果。
因此，两个全连接层可以帮助模型尽可能地利用输入序列中的信息，从而提高预测性能。总之，一维卷积层、扁平化层和全连接层一起可以构成一个强大的LSTM模型，可以处理序列数据并进行准确的预测。
"""

def bp_lstm(lstm_units, dense1, look_back):
    inputs = Input(shape=(look_back, INPUT_DIMS))       # INPUT_DIMS：输入序列中每个时间步的特征数
                                                                                         # inputs：模型的输入张量，形状为(look_back, INPUT_DIMS)
    x = Conv1D(filters=128, kernel_size=1, activation='relu')(inputs)  # 定义一个卷积层，它对输入序列进行滑动窗口卷积操作，并使用ReLU激活函数激活输出 , padding = 'same'
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, activation='relu'), name='bilstm')(x)  # 定义一个双向LSTM层，它接收卷积层的输出作为输入，并返回整个序列的隐藏状态序列
    attention_mul = attention_3d_block(lstm_out)                # 一个注意力机制层，它接收双向LSTM的输出作为输入，并返回注意力加权后的输出
    attention_mul = Flatten()(attention_mul)                        # 一个扁平化层，它将注意力加权后的输出展平成一维张量
    output = Dense(dense1)(attention_mul)                          # 第一个全连接层，它接收扁平化后的注意力加权输出作为输入，并输出一个具有dense1个神经元的向量
    output = Dense(pre_year, activation='linear')(output)       # 最后一个全连接层，它接收第一个全连接层的输出作为输入，并输出一个具有pre_year个神经元的向量，使用线性激活函数。
    model = Model(inputs=[inputs], outputs=output)              # 根据输入和输出张量创建一个Keras模型。
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
train_path = 'B0005.csv'
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
INPUT_DIMS = train.shape[1] # shape 函数返回一个元组，其中第一个元素是数据框的行数，第二个元素是列数
# 预测未来1次的容量
pre_year = 1


# 调参方法
def objective(trial):
    lr = trial.suggest_uniform('lr', 0.002, 0.1)
    batch_size = trial.suggest_int('batch_size', 4, 6)
    lstm_units = trial.suggest_int('lstm_units', 33, 55)
    dense1 = trial.suggest_int('dense1', 54, 99)
    epochs = trial.suggest_int('epochs', 10, 100)
    look_back = trial.suggest_int('look_back',1, 2)
    train_X, train_Y = create_dataset(train_x, train_y, look_back)
    train_Y_data = pd.DataFrame(train_Y)
    train_Y_data.dropna(inplace=True)
    train_X = train_X[:train_Y_data.shape[0]]
    train_Y = train_Y_data.values
    train_X = train_X.astype('float64')
    m = bp_lstm(lstm_units, dense1, look_back)
    optimizer = Adam(lr)
    m.compile(loss='mae', optimizer=optimizer)
    hist = m.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=0)
    mae = hist.history['loss'][-1]
    return mae


""" 
把这一段反注释掉就可以开启调参模型，command + /
"""
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('Best trial number: ', study.best_trial.number)
print('Best value:', study.best_trial.value)  # 0.04
print('Best parameters: \n', study.best_trial.params)
parameters = study.best_trial.params
# 原来参数
# parameters = {'lr': 0.01782551732488114, 'batch_size': 6, 'lstm_units': 43, 'dense1': 83, 'epochs': 50, 'look_back': 2}
#当前最优参数
# parameters = {'lr': 0.0020629683964801075, 'batch_size': 5, 'lstm_units': 43, 'dense1': 83, 'epochs': 79, 'look_back': 2} #0.006535541096296771
parameters =   {'lr': 0.002247088379699024, 'batch_size': 5, 'lstm_units': 45, 'dense1': 82, 'epochs': 57, 'look_back': 2}

lstm_units = parameters['lstm_units']
lr = parameters['lr']
look_back = parameters['look_back']
batch_size = parameters['batch_size']
dense1 = parameters['dense1']
epochs = parameters['epochs']

train_X, train_Y = create_dataset(train_x, train_y, look_back)
m = bp_lstm(lstm_units, dense1, look_back)
optimizer = Adam(lr)
m.compile(loss='mae', optimizer=optimizer)
hist = m.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size)

test_X, test_Y = create_dataset(test_x, test_y, look_back)
test_X = test_X.astype('float64')
pred_y = m.predict(test_X)
mae = mean_absolute_error(test_Y, pred_y)
rmse = mean_squared_error(test_Y, pred_y)**0.5
print("MAE = {} ".format(mae))
print("RMSE = {}".format(rmse))

inv_pred_y = test_y_MinMax.inverse_transform(pred_y.reshape(-1, 1))

# 下面内容做到容量计算SOH，使用原始数据——不加入可监测参数，使用可监测参数数据由于有SOH属性，故不需要下面的操作。
inv_pred_y = inv_pred_y / 2
df.iloc[:, 0] = df.iloc[:, 0] / 2       #df.iloc[:,0]表示df对象的所有第一列数据

# 画预测和真实值的对比图
# plt.plot([x for x in range(train_size, 50282)], inv_pred_y, color="red", label="pred")
# plt.plot([x for x in range(50285)], df.values[:, 0], color="blue", label="pred")
plt.plot([x for x in range(train_size, 164)], inv_pred_y, color="red", label="pred")
plt.plot([x for x in range(167)], df.values[:, 0], color="blue", label="pred")

plt.ylabel('锂电池SOH')
plt.xlabel('放电状态监测次数')
plt.legend(['SOH预测值', 'SOH真实值'], loc='best')
# 绘图
plt.title('B0005_CNN+BiLSTM+Attention_SOH预测')
plt.savefig("SOH_CNN+BiLSTM+Attention_预测.png", dpi=500, bbox_inches='tight')
# 反注释后，画出使用Adam后的运行结果，需要把原来的绘图代注释掉
'''
plt.title('B0005_CNN+BiLSTM+Attention_Adam_SOH预测')
plt.savefig("B0005_CNN+BiLSTM+Attention_Adam_SOH预测.png", dpi=500, bbox_inches='tight')
'''
plt.show()
