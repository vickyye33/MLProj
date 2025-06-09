import pickle
from typing import Any

import arrow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
# TODO:[-] 25-06-03 keras.src 路径是 TensorFlow 2.11 及更高版本中集成在 TensorFlow 内部的 Keras 3 中使用的。
# from keras.src.layers import LSTM, Dropout, Bidirectional, Dense, Masking
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Masking
from pandas import DatetimeIndex
from keras.models import load_model
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


def load_customer_model(model_path: str) -> Any:
    if pathlib.Path(model_path).exists():
        loaded_model = load_model(model_path)
        """加载后的模型"""
        print(loaded_model.summary())
        return loaded_model
    return None


def load_customer_scaler(model_path: str) -> Any:
    if pathlib.Path(model_path).exists():
        loaded_scaler = pickle.load(model_path)
        return loaded_scaler
    return None


def main():
    model_path: str = r'E:\05DATA\fit_model_v2_250609.h5'
    loaded_model = load_customer_model(model_path)

    # TODO:[-] 25-06-08 新加入的razer配置
    forecast_path: str = r'Z:\SOURCE_MERGE_DATA\df_ws_forecast.csv'
    realdata_path: str = r'Z:\SOURCE_MERGE_DATA\2024_local_df_utc_183_split.csv'
    scaler_forecast_path: str = r'Z:\01TRAINNING_DATA\scaler\scaler_forecast_250609.sav'
    scaler_realdata_path: str = r'Z:\01TRAINNING_DATA\scaler\scaler_realdata_250609.sav'

    # step1: 加载标准化后的 预报 | 实况 数据集
    df_forecast = pd.read_csv(forecast_path, encoding='utf-8', index_col=0)
    df_realdata = pd.read_csv(realdata_path, encoding='utf-8', index_col=0)
    scaler_forecast = load_customer_scaler(scaler_forecast_path)
    scaler_realdata = load_customer_scaler(scaler_realdata_path)

    # 实况拼接有问题需要手动去掉最后一列
    # df_realdata = df_realdata.drop(df_realdata.columns[-1], axis=1)
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

    X_scaled = scaler_forecast.transform(X)
    y_scaled = scaler_realdata.transform(y)

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

    # step4: 模型预测
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)

    # 注意由于 model.add(Dense(25)) 加入了全连接层，最后一步对每个时间输出25维结果，所以暂时取出第一个维度的数据
    y_pred = loaded_model.predict(X_test)
    y_pred = y_pred[:, :, 0]

    # 反归一化
    # ERROR: ValueError: Found array with dim 3. None expected <= 2.
    y_pred_real = y_scaled.inverse_transform(y_pred)
    y_test = y_test[:, :, 0]
    y_test_real = y_scaled.inverse_transform(y_test)
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    # 真实数据上的 RMSE: 4.6269
    print(f'真实数据上的 RMSE: {rmse_real:.4f}')
    pass


if __name__ == '__main__':
    main()
