import arrow
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
# TODO:[-] 25-06-03 keras.src 路径是 TensorFlow 2.11 及更高版本中集成在 TensorFlow 内部的 Keras 3 中使用的。
# from keras.src.layers import LSTM, Dropout, Bidirectional, Dense, Masking
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Masking
from pandas import DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.losses import MeanSquaredError
# 可视化结果（如果需要）
import matplotlib.pyplot as plt
import os
import pathlib
import xml.etree.ElementTree as ET
import xarray as xr
import codecs
import datetime

# 先从海浪数据中提取出经纬度，时间，风，海浪高度
# 解析单个文件，并存于字典内
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# from utils import rmse


def to_do():
    """

    :return:
    """
    # vals
    # mac
    # forecast_path: str = r'/Volumes/DATA/01TRAINNING_DATA/WIND/01/df_ws_forecast.csv'
    # realdata_path: str = r'/Volumes/DATA/FUB/MF01001/2024_local_df_utc_filter.csv'
    # win
    # forecast_path: str = r'Z:\01TRAINNING_DATA\standard\df_ws_forecast.csv'
    # realdata_path: str = r'Z:\01TRAINNING_DATA\standard\2024_local_df_utc_filter.csv'
    # model_path: str = r'G:\05DATA\02MODELS\fit_model_250603.h5'
    # razer
    # TODO:[-] 25-06-08 新加入的razer配置
    forecast_path: str = r'Z:\SOURCE_MERGE_DATA\df_ws_forecast.csv'
    realdata_path: str = r'Z:\SOURCE_MERGE_DATA\2024_local_df_utc_183_split.csv'
    model_path: str = r'E:\05DATA\fit_model_v2_250612.h5'
    scaler_forecast: str = r'Z:\01TRAINNING_DATA\scaler\scaler_forecast_250609.sav'
    scaler_realdata: str = r'Z:\01TRAINNING_DATA\scaler\scaler_realdata_250609.sav'

    # step1: 加载标准化后的 预报 | 实况 数据集
    # shape: (61,732)
    df_forecast = pd.read_csv(forecast_path, encoding='utf-8', index_col=0)
    # shape:(61,731)
    df_realdata = pd.read_csv(realdata_path, encoding='utf-8', index_col=0)
    # 实况拼接有问题需要手动去掉最后一列
    # 使用按3小时进行分割的数据不需要去掉最后一列，这样 realdata 与 forecast 的 columns 一致
    # df_realdata = df_realdata.drop(df_realdata.columns[-1], axis=1)
    df_forecast = df_forecast.iloc[:61, :]
    print(f'df_forecast.shape: {df_forecast.shape}')
    print(f'df_realdata.shape: {df_realdata.shape}')
    pass
    # step2: 由于数据中存在nan，如何处理nan
    pass
    # step3: 数据标准化(提出nan值)
    rows: int = df_forecast.shape[0]
    cols: int = df_forecast.shape[1]
    # TODO:[-] 25-05-28 注意原始数据中: forecast (61,732), real (61,732)
    X = df_forecast.values.T.reshape(cols, rows, 1)
    # TODO:[*] 25-05-11 注意 y 中有存在 nan
    # ValueError: cannot reshape array of size 52776 into shape (732,72,1)
    y = df_realdata.values.T.reshape(cols, rows, 1)

    # step3-2:对数据进行归一化
    # 拍扁数据为二维数组（n*timesteps, feature）进行归一化
    X_flat = X.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)

    # 分别为 X 和 y 定义归一化器（当然如果两者量纲一致，可用同一个 scaler）
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_X.fit_transform(X_flat)
    y_scaled = scaler_y.fit_transform(y_flat)
    # TODO:[-] 25-06-09 保存归一化器
    joblib.dump(scaler_X, scaler_forecast)
    joblib.dump(scaler_y, scaler_realdata)

    # 将归一化后的二维数据恢复为原来的3D形状
    X = X_scaled.reshape(X.shape)
    y = y_scaled.reshape(y.shape)

    # step3-3: 对数据集进行划分
    # 拆分数据集为训练集和测试集
    # 此处 *_tran 均为 (585,72,1) | *_test 均为 (147,72,1)
    # X_* 相当于是 预报数据集 | y_* 是实况(验证)数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 此处加入转换

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # step4: 构建模型
    model = Sequential()
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)
    """
        错误原因:
            你在 Masking 层中设置了 input_shape=(25, 1)，这表示你期望输入数据的每个样本有 25 个时间步，每个时间步有 1 个特征。
            然而，错误信息 found shape=(None, 72, 1) 清楚地表明，你的实际输入数据 X_train 和 X_test 的每个样本有 72 个时间步，而不是 25 个。
            TODO:[-] 25-06-0 此处做了修改由于数据为 (61,732) 故 72 => 61

    """
    # 模型构建的步骤:
    # step1: 添加一个 Masking 层，指定输入中值为 0.0 的时刻将被“屏蔽”，即这些时间步不会对后续层产生影响。
    # TODO:[-] 25-06-12 屏蔽是会去掉该时刻的所有数据吗？
    model.add(Masking(mask_value=0.0, input_shape=(61, 1)))
    # step2: 添加双向LSTM层
    # 双向 LSTM 同时从前向和后向处理时序数据，从而捕获更多上下文信息，提升特征提取能力。
    # units=128：LSTM 层中每个方向上有 128 个神经元。
    # return_sequences=True：输出每个时间步的结果，而非仅仅输出最后时刻的状态，这样可以将整个序列的信息传递到下一层。
    # activation='relu'：将激活函数设置为 ReLU（而非 LSTM 默认的 tanh），可能有助于缓解梯度消失问题，不过这取决于具体任务。
    # 注意：虽然此处指定了 input_shape=(25, 1)，但实际上在 Sequential 模型中第一层已经指定了输入形状，所以这里的 input_shape 参数可能是不必要或引起混淆（建议保持与 Masking 层一致，即 (61, 1)）。
    # v1 激活函数:relu
    # v2 改为: tanh
    model.add(Bidirectional(LSTM(units=256, return_sequences=True,
                                 activation='tanh',
                                 input_shape=(61, 1))))
    # step3: 添加 Dropout 层，在训练时随机将 20% 的神经元输出设为 0。
    # Dropout 是一种正则化方法，有助于防止模型过拟合；通过随机丢弃部分神经元，模型不能过分依赖局部特征，从而提高泛化能力。
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, activation='tanh')))  # 可以堆叠多个LSTM层
    # step5: 再次添加一个 Dropout 层，使得第二层 LSTM 的输出在训练时有 20% 被随机置零，从而进一步防止过拟合。
    model.add(Dropout(0.2))
    # step4: 第二层双向LSTM层
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, activation='tanh')))  # 可以堆叠多个LSTM层
    # step5: 再次添加一个 Dropout 层，使得第二层 LSTM 的输出在训练时有 20% 被随机置零，从而进一步防止过拟合。
    model.add(Dropout(0.2))
    # step6: 添加全连接层（Dense 层），输出节点数为 25。25-06-12 此处修改为 61 ，需要预测长度为61的时间向量
    # Dense 层将前面的时序特征映射到目标空间。在时序网络中，当上层返回整个序列（形状为 [batch_size, time_steps, features]）时，Dense 层会被逐时步地应用，输出每个时间步对应一个长度为 25 的向量。这通常用于预测任务，比如多步预测或者每个时刻有多个目标值的任务。
    model.add(Dense(61))

    # 将均方误差修改为均方根误差后
    # step7: 编译模型
    # optimizer='adam' 使用 Adam 优化器进行梯度下降更新。Adam 优化器能够自适应调整各参数的学习率，通常能较快收敛，并且对超参数设定不太敏感，不同于传统的 SGD。
    # TODO:[*] 25-06-12 什么是超参数？
    # loss='mse' 将损失函数设为均方误差（Mean Squared Error），这是回归问题常用的误差度量指标，模型训练的目标就是尽可能使预测值与真实值之间的均方误差最小化。
    # 整体作用 model.compile() 会对模型进行配置，指定训练时用哪个优化器、用哪个损失函数，如果需要，还可以添加额外的评估指标。编译过程会为模型建立必要的计算图，并对参数进行初始化。
    model.compile(optimizer='adam', loss='mse')

    # step8: 训练模型
    # X_train, y_train 分别为训练数据和对应目标值。模型将以这些数据为依据不断调整参数，使预测值与真实目标值之间的 MSE 最小化。
    # epochs=10 指定训练过程需要遍历整个训练集 10 次。每个 epoch 内部数据会根据 batch 大小分批更新参数。
    # batch_size=16 每个训练步骤（step）使用 16 个样本进行梯度计算和模型更新。较小的 batch size 有助于捕捉更多细微变化，但训练时间可能更长；较大的 batch size 则计算稳定但可能导致泛化性下降。
    # validation_data=(X_test, y_test) 在每个 epoch 结束后，模型也会评估一次在验证集上的损失。这有助于监控过拟合情况以及训练过程的稳定性。
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    model.save(model_path)

    pass

    pass


def main():
    to_do()


if __name__ == '__main__':
    main()
