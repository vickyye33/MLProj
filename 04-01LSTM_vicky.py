import arrow
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
import joblib

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
    forecast_path: str = r'Z:\01TRAINNING_DATA\standard\df_ws_forecast.csv'
    realdata_path: str = r'Z:\01TRAINNING_DATA\standard\2024_local_df_utc_filter.csv'
    model_path: str = r'G:\05DATA\02MODELS\fit_model_250606.h5'
    scaler_X_path: str = r'G:\05DATA\02MODELS\scaler_X.pkl'
    scaler_y_path: str = r'G:\05DATA\02MODELS\scaler_y.pkl'

    # step1: 加载标准化后的 预报 | 实况 数据集
    df_forecast = pd.read_csv(forecast_path, encoding='utf-8', index_col=0)
    df_realdata = pd.read_csv(realdata_path, encoding='utf-8', index_col=0)
    # 实况拼接有问题需要手动去掉最后一列
    df_realdata = df_realdata.drop(df_realdata.columns[-1], axis=1)
    print(f'df_forecast.shape: {df_forecast.shape}')
    print(f'df_realdata.shape: {df_realdata.shape}')
    pass
    # step2: 由于数据中存在nan，如何处理nan
    pass
    # step3: 数据标准化(提出nan值)
    rows: int = df_forecast.shape[0]
    cols: int = df_forecast.shape[1]
    # TODO:[-] 25-05-28 注意原始数据中: forecast (72,732), real (72,733)
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

    # 保存归一化器
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

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

    """
    model.add(Masking(mask_value=0.0, input_shape=(72, 1)))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True,
                                 activation='relu',
                                 input_shape=(25, 1))))  # units是LSTM神经元数量, return_sequences=True 因为我们要在每个时间步都输出
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, activation='relu')))  # 可以堆叠多个LSTM层
    model.add(Dropout(0.2))
    model.add(Dense(25))

    # 编译模型
    # TODO:[-] 25-05-14 此处损失函数使用 RMSE——均方根误差
    # 将均方误差修改为均方根误差后

    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    # ERROR:
    # TypeError: Cannot interpret 'tf.float32' as a data type
    # Epoch 1/10
    # 2025-05-29 20:13:55.381747: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
    #  2/37 ━━━━━━━━━━━━━━━━━━━━ 8:18 14s/step - loss: nan
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    model.save(model_path)

    pass

    pass


def main():
    to_do()


if __name__ == '__main__':
    main()
