import csv
from typing import Optional

import arrow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src.layers import LSTM, Dropout, Bidirectional, Dense
from pandas import DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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

from utils import rmse


def batch_get_realdata(file_full_path: str, split_hours=72, issue_hours_steps: int = 12) -> pd.DataFrame:
    """
        TODO:[-] 25-04-23 生成实况训练数据集
        从指定文件批量获取时间数据并以dataframe的形式返回
    :param file_full_path:
    :return:
    """

    """
        eg: csv文件样例:
                        time	longitude	latitude	WS	YBG
                        202401010000
                        YYYYMMDDHHmm
    """
    list_series = []
    merge_dict = {}
    if pathlib.Path(file_full_path).exists():
        # ds: xr.Dataset = xr.open_dataset(file_full_path)
        df: pd.DataFrame = pd.read_csv(file_full_path)
        """读取指定路径的浮标处理后的一年的数据"""
        # 通过起止时间找到对应的index，然后每次的发布时间间隔步长为12h

        # step1: 生成2024年一年的时间步长为1hour的时间索引集合
        # TODO:[-] 25-05-12 注意此部分修改起止时间 local time -> utc time
        start_time = '2024-01-01 00:00:00'
        end_time = '2024-12-31 23:00:00'
        start_time_utc = '2023-12-31 16:00:00'
        end_time_utc = '2024-12-31 15:00:00'
        time_series = pd.date_range(start=start_time, end=end_time, freq='H')

        # 将time列的内容从int64 => str
        df['time'] = df['time'].astype(str)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
        # step2: 将 time列设置为index，并将index替换为标准时间集合
        df.set_index('time', inplace=True)
        # TODO:[-] 25-05-12 注意直接重设index会造成所有数据为 nan 的情况。 解决办法：先创建新的utc_times 再设置为索引列
        # ValueError: Length of values (8784) does not match length of index (8887)

        df_reindexed = df.reindex(time_series)
        utc_times = pd.date_range(start=start_time_utc, end=end_time_utc, freq='H', tz='utc')
        df_reindexed['utc_times'] = utc_times
        df_reindexed = df_reindexed.set_index('utc_times')
        # 此处不需要再删除 time了，因为 index 已经 utc_times -> index了
        # df_reindexed = df_reindexed.drop('time', axis=1)
        df_reindexed.index.name = 'time'

        # step3: 生成12小时为间隔的时间数组
        freq_str: str = f'{issue_hours_steps}H'
        start_time_split_utc: str = '2024-01-01 00:00:00'
        end_time_split_utc: str = '2024-12-31 23:00:00'
        issue_dt_series = pd.date_range(start=start_time_split_utc, end=end_time_split_utc, freq=freq_str, tz='utc')

        for temp_time in issue_dt_series:
            temp_index: int = df_reindexed.index.get_loc(temp_time)
            val_series = df_reindexed[temp_index:temp_index + split_hours]
            list_series.append(val_series)
        # TODO:[-] 25-04-24 此处做重新修改，拼接成一个dataframe

        for temp_time in issue_dt_series:
            dt_str: str = temp_time.strftime('%Y%m%d%H%M%S')
            temp_index: int = df_reindexed.index.get_loc(temp_time)
            val_series = df_reindexed[temp_index:temp_index + split_hours]
            # 此处改为只取 'WS' 列
            # TODO:[-] 25-04-24 住一次此处需要将每一个 series的index索引重置为 [0,71]
            merge_dict[dt_str] = val_series['WS'].reset_index(drop=True)
            # list_series.append(val_series)
    df = pd.DataFrame.from_dict(merge_dict)
    # TODO:[-] 25-05-11 注意此处的 columns 为 字符串，需要将 str -> datetime
    df.columns = pd.to_datetime(df.columns)
    return df


def get_test_array(test_read_path: str, training_read_path: str, issue_times_index: DatetimeIndex):
    """
        分别读取测试数据集以及实况数据集并进行训练
    :param test_read_path:
    :param training_read_path:
    :return:
    """
    if pathlib.Path(test_read_path).exists() and pathlib.Path(training_read_path).exists():
        df_test: pd.DataFrame = pd.read_csv(test_read_path)
        u_data_dict = {}
        v_data_dict = {}
        # 读取的预报风场——测试训练集 在 df 中是通过 xxx_u与 xxx_v 的形式进行存储
        # TODO:[-] 25-04-28 u 与 v 每个共613组预报数据
        for col_name in df_test.columns:
            try:
                col_vector = df_test[col_name]
                # yyyymmddhhss
                dt_temp_str: str = col_name.split('_')[0]
                # u or v
                var_temp_str: str = col_name.split('_')[1]
                if var_temp_str == 'u':
                    # u_data_dict[dt_temp_str] = col_vector.tolist()
                    u_data_dict[dt_temp_str] = col_vector
                elif var_temp_str == 'v':
                    # v_data_dict[dt_temp_str] = col_vector.tolist()
                    v_data_dict[dt_temp_str] = col_vector
                print(f'当前列:{col_name}处理成功~')
            except Exception as e:
                print(f'当前列:{col_name}处理错误!')
        # # step2: 将字典统一转换为二维数组
        # result_u_array = [val for key, val in u_data_dict.items()]
        # result_v_array = [val for key, val in v_data_dict.items()]
        # return [result_u_array, result_v_array]
        df_u = pd.DataFrame.from_dict(u_data_dict)
        df_v = pd.DataFrame.from_dict(v_data_dict)
        # 将时间字符串=>datetime
        df_u.columns = pd.to_datetime(df_u.columns)
        df_v.columns = pd.to_datetime(df_v.columns)
        # TODO:[*] 25-04-29
        # 需要根据起止时间及时间步长，生成对应的时间索引，并将该时间索引作为标准索引
        # 注意： reindex 后会返回一个新的 DataFrame，并不会修改原始df
        df_u = df_u.reindex(columns=issue_times_index)
        df_v = df_v.reindex(columns=issue_times_index)
        return df_u, df_v
        # pass
    return None


def main_backup():
    """
        25-05-08
    :return:
    """
    read_path: str = r'Z:\WIND\GRAPES\2024'
    out_put_path: str = r'./data'
    out_put_file_path: str = str(pathlib.Path(out_put_path) / 'GRAPES_2024_24')
    lat: float = 39.5003
    lng: float = 120.59533

    # TODO:[*] 25-04-29 根据起止时间，时间步长为12h，生成发布时间index
    start_time = '2024-01-01 00:00:00'
    end_time = '2024-12-31 23:00:00'
    issue_times_index = pd.date_range(start=start_time, end=end_time, freq='12h')

    # step3: 提取 test 与 training data 开始训练
    # traning_ws(r'G:\05DATA\01TRAINING_DATA\WIND\merge.csv', r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv')

    """shape:(25,732)"""
    df_u, df_v = get_test_array(r'G:\05DATA\01TRAINING_DATA\WIND\merge.csv',
                                r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv', issue_times_index)

    read_file_full_path: str = r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv'
    # step1: 生成一年的 365*2 =730 个 ws,ybg -> 只取 ws
    df_ws = batch_get_realdata(read_file_full_path)
    """shape:(72,732)"""

    # TODO:[-] 25-05-08 将 u+v获得 ws的风速的值
    df_ws_forecast = np.sqrt(df_u ** 2 + df_v ** 2)
    """风场的风速绝对值"""

    # shape :(25,732)
    df_ws_forecast = pd.DataFrame(df_ws_forecast)

    # 将 DataFrame 转换为 NumPy 数组
    X = []
    y = []

    # 设置时间步长为 25
    n_timesteps = 24
    # 25 相当于样本数也是25
    n_samples = df_ws_forecast.shape[0]

    """
        当 n_timesteps 设置为 24 时，你可以从 25 个时间点中生成 1 个有效样本：
        使用 df_ws_forecast 的前 24 个时间点作为输入 X。
        使用第 25 个时间点的 df_ws 作为输出 y。
    """
    # 构建输入输出数据
    for i in range(n_timesteps, n_samples):
        # df_ws_forecast.values[24-24:24,1:]
        # 获取 第 i-n 行 -> i 行，所有列的数据
        X.append(df_ws_forecast.values[i - n_timesteps:i, :])  # 过去 25 小时的预报
        # df_ws.vlaues[24,:]
        y.append(df_ws.values[i, :])  # 当前时间点的实际观测值

    X = np.array(X)
    y = np.array(y)

    # 数据形状
    print("X shape:", X.shape)  # 应为 (样本数, 25, 732) | 实际: (1,24,732)
    print("y shape:", y.shape)  # 应为 (样本数, 732)     | 实际: (1,732)

    # 拆分数据集为训练集和测试集
    # TODO:[*] 25-05-08
    #  ValueError: With n_samples=1, test_size=0.2 and train_size=None, the resulting train set will be empty.
    #  Adjust any of the aforementioned parameters.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建 LSTM 模型
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))  # 防止过拟合
    model.add(Dense(732))  # 输出层，732 个风速预测值

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

    # 评估模型
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    # 进行预测
    predictions = model.predict(X_test)

    # # 可视化结果（如果需要）
    # import matplotlib.pyplot as plt
    #
    # # 选择第一个样本进行可视化
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test[0], label='Actual', color='blue')
    # plt.plot(predictions[0], label='Forecast', color='red')
    # plt.title('Wind Speed Prediction')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Wind Speed')
    # plt.legend()
    # plt.show()
    pass


def main():
    """
        25-05-11
    :return:
    """
    read_path: str = r'Z:\WIND\GRAPES\2024'
    out_put_path: str = r'./data'
    out_put_file_path: str = str(pathlib.Path(out_put_path) / 'GRAPES_2024_24')
    lat: float = 39.5003
    lng: float = 120.59533

    # TODO:[*] 25-04-29 根据起止时间，时间步长为12h，生成发布时间index
    start_time = '2024-01-01 00:00:00'
    end_time = '2024-12-31 23:00:00'
    issue_times_index = pd.date_range(start=start_time, end=end_time, freq='12h')

    # step1: 提取数据集
    df_u, df_v = get_test_array(r'G:\05DATA\01TRAINING_DATA\WIND\merge.csv',
                                r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv', issue_times_index)

    read_file_full_path: str = r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv'
    #  生成一年的 365*2 =730 个 ws,ybg -> 只取 ws
    # 注意实况 shape : (72,732)
    df_ws = batch_get_realdata(read_file_full_path)
    """shape:(72,732)"""

    df_ws_subset = df_ws.iloc[:25, :]

    # TODO:[-] 25-05-08 将 u+v获得 ws的风速的值
    df_ws_forecast = np.sqrt(df_u ** 2 + df_v ** 2)
    """风场的风速绝对值"""

    # shape :(25,732)
    df_ws_forecast = pd.DataFrame(df_ws_forecast)

    print(f'df_ws_forecast shape:{df_ws_forecast.shape}')
    print(f'df_ws_subset shape:{df_ws_subset.shape}')
    """
        df_ws_forecast shape:(25, 732)
        df_ws_subset shape:(25, 732)
    """

    # TODO:[*] 25-05-11 由于数据可能存在空置（大概40个预报时次，20天没有预报数据)
    # 删除 df_ws_forecast 中的空值，并获取有效索引
    # axis=1 按列检索，若列中存在nan则删除该列
    valid_columns = df_ws_forecast.dropna(axis=1).columns

    # 使用有效索引筛选 df_ws_forecast 和 df_ws_subset
    df_ws_forecast_cleaned = df_ws_forecast[valid_columns]
    df_ws_subset_cleaned = df_ws_subset[valid_columns]
    print(f'df_ws_forecast_cleaned shape:{df_ws_forecast_cleaned.shape}')
    print(f'df_ws_subset_cleaned shape:{df_ws_subset_cleaned.shape}')
    print(f'df_ws_forecast_cleaned中包含 NaN 的行数:{df_ws_forecast_cleaned.isnull().any(axis=0).sum()}')
    print(f'df_ws_subset_cleaned NaN 的行数:{df_ws_subset_cleaned.isnull().any(axis=0).sum()}')
    print('# ————————————————————————————————————————————')
    # ————————————————————————————————————————————

    # 使用线性插值填充 NaN 值
    df_interpolated = df_ws_subset_cleaned.interpolate(method='linear')
    print(f'df_interpolated中包含 NaN 的行数:{df_interpolated.isnull().any(axis=0).sum()}')
    print('# ————————————————————————————————————————————')
    # 由于实况数据中也可能包含nan，需要找到实况数据中的nan的行，并双向删除
    # TODO:[-] 25-05-11
    # axis=0 ，表示按行进行操作 ; axis=1，表示按列进行操作 .
    valid_realdata_columns = df_interpolated.dropna(axis=1).columns
    df_ws_forecast_cleaned = df_ws_forecast[valid_realdata_columns]
    df_ws_subset_cleaned = df_interpolated[valid_realdata_columns]
    print(f'df_ws_forecast_cleaned shape:{df_ws_forecast_cleaned.shape}')
    print(f'df_ws_subset_cleaned shape:{df_ws_subset_cleaned.shape}')
    print(f'df_ws_forecast_cleaned中包含 NaN 的行数:{df_ws_forecast_cleaned.isnull().any(axis=0).sum()}')
    print(f'df_ws_subset_cleaned NaN 的行数:{df_ws_subset_cleaned.isnull().any(axis=0).sum()}')
    print('# ————————————————————————————————————————————')
    """
        df_ws_forecast_cleaned shape:(25, 612)
        df_ws_subset_cleaned shape:(25, 612)
    """
    # ————————————————————————————————————————————

    # step2: 数据格式化
    # 将 DataFrame 转换为 NumPy 数组
    X = []
    y = []

    # 再次检查清理后的df是否还存在空值
    # 检查 df_ws_forecast 是否有缺失值
    # cleaned_forecast = df_ws_forecast_cleaned.isnull().sum()
    # cleaned_realdata = df_ws_subset_cleaned.isnull().sum()
    # print("Missing values in df_ws_forecast_cleaned:\n", cleaned_forecast)
    # print("Missing values in df_ws_subset_cleaned:\n", cleaned_realdata)

    rows: int = df_ws_forecast_cleaned.shape[0]
    cols: int = df_ws_forecast_cleaned.shape[1]

    X = df_ws_forecast_cleaned.values.T.reshape(cols, rows, 1)
    # TODO:[*] 25-05-11 注意 y 中有存在 nan
    y = df_ws_subset_cleaned.values.T.reshape(cols, rows, 1)
    # TODO:[-] 25-05-11 由于输出结果均为 nan，排查输入的 X 与 y 是否存在 nan
    if np.isnan(X).any() or np.isnan(y).any():
        print("数据中存在 NaN 值，请处理。")

    print(f'df_ws_forecast.T-> X shape:{X.shape}')
    print(f'df_ws_subset.T-> y shape:{y.shape}')
    """
        df_ws_forecast.T-> X shape:(732, 25, 1)
        df_ws_subset.T-> y shape:(732, 25, 1)
    """

    # ============================
    # 归一化步骤
    # ============================
    # 拍扁数据为二维数组（n*timesteps, feature）进行归一化
    X_flat = X.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)

    # 分别为 X 和 y 定义归一化器（当然如果两者量纲一致，可用同一个 scaler）
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_X.fit_transform(X_flat)
    y_scaled = scaler_y.fit_transform(y_flat)

    # 将归一化后的二维数据恢复为原来的3D形状
    X = X_scaled.reshape(X.shape)
    y = y_scaled.reshape(y.shape)

    # 设置时间步长为 25
    n_timesteps = 24
    # 25 相当于样本数也是25
    n_samples = df_ws_forecast_cleaned.shape[0]

    # step3: 数据划分
    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # step4: 创建 LSTM 模型
    model = Sequential()
    # 25 时间步长，1个特征
    # 模型1:
    # model.add(LSTM(128, activation='relu', input_shape=(25, 1)))
    # # TODO:[*] 25-05-11 此处的意义
    # model.add(Dropout(0.2))  # 防止过拟合
    # # ValueError: Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 100)
    # model.add(LSTM(64, activation='relu'))
    # model.add(Dropout(0.2))  # 防止过拟合
    # model.add(Dense(25))  # 输出层，预测 25 个时间步的值

    # 模型2:

    # model = Sequential()
    model.add(Bidirectional(LSTM(units=128, return_sequences=True,
                                 activation='relu',
                                 input_shape=(25, 1))))  # units是LSTM神经元数量, return_sequences=True 因为我们要在每个时间步都输出
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, activation='relu')))  # 可以堆叠多个LSTM层
    model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(units=64, return_sequences=True, activation='relu')))  # 可以堆叠多个LSTM层
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(units=64, return_sequences=True, activation='relu')))  # 可以堆叠多个LSTM层
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(units=32, return_sequences=True, activation='relu')))  # 可以堆叠多个LSTM层
    # model.add(Dropout(0.2))
    model.add(Dense(25))

    # 编译模型
    # TODO:[-] 25-05-14 此处损失函数使用 RMSE——均方根误差
    # 将均方误差修改为均方根误差后
    # 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - loss: 2.1976
    # Test Loss: 2.200413465499878
    # model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='adam', loss=rmse)

    # 训练模型
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

    # step5: 模型预测
    # 进行预测
    predictions = model.predict(X_test)
    # 输出预测结果的形状
    print("Predictions shape:", predictions.shape)  # 应为 (147, 25)

    # 如果 Dense 层输出二维数据（形如 (n_samples, 25)），则增加一个维度以方便后续反归一化
    if predictions.ndim == 2:
        predictions = np.expand_dims(predictions, axis=-1)
    if y_test.ndim == 2:
        y_test = np.expand_dims(y_test, axis=-1)

    # ============================
    # 反归一化
    # ============================
    # 将预测结果与对应标签拍扁为二维数组，方便调用 scaler.inverse_transform
    pred_flat = predictions.reshape(-1, 1)
    y_test_flat = y_test.reshape(-1, 1)

    # 用 scaler_y 将归一化数值恢复到原始尺度
    pred_inv = scaler_y.inverse_transform(pred_flat)
    y_test_inv = scaler_y.inverse_transform(y_test_flat)

    # 恢复反归一化后的形状与 original predictions 保持一致
    pred_inv = pred_inv.reshape(predictions.shape)
    y_test_inv = y_test_inv.reshape(y_test.shape)

    # ============================
    # 计算均方误差 (MSE)
    # ============================
    mse_value = mean_squared_error(y_test_inv.flatten(), pred_inv.flatten())
    print(f'Test MSE: {mse_value}')

    # 评估模型
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    # 选择第一个样本进行可视化
    # import matplotlib
    # matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test[0], label='Actual', color='blue')
    # plt.plot(predictions[0], label='Forecast', color='red')
    # plt.title('Wind Speed Prediction')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Wind Speed')
    # plt.legend()
    # plt.show()
    pass


def main_250515():
    """
        25-05-11
    :return:
    """
    read_path: str = r'Z:\WIND\GRAPES\2024'
    out_put_path: str = r'./data'
    out_put_file_path: str = str(pathlib.Path(out_put_path) / 'GRAPES_2024_24')
    lat: float = 39.5003
    lng: float = 120.59533

    # TODO:[*] 25-04-29 根据起止时间，时间步长为12h，生成发布时间index
    start_time = '2024-01-01 00:00:00'
    end_time = '2024-12-31 23:00:00'
    issue_times_index = pd.date_range(start=start_time, end=end_time, freq='12h')

    # step1: 提取数据集
    df_u, df_v = get_test_array(r'G:\05DATA\01TRAINING_DATA\WIND\merge.csv',
                                r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv', issue_times_index)

    read_file_full_path: str = r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv'
    #  生成一年的 365*2 =730 个 ws,ybg -> 只取 ws
    # 注意实况 shape : (72,732)
    df_ws = batch_get_realdata(read_file_full_path)
    """shape:(72,732)"""

    df_ws_subset = df_ws.iloc[:25, :]

    # TODO:[-] 25-05-08 将 u+v获得 ws的风速的值
    df_ws_forecast = np.sqrt(df_u ** 2 + df_v ** 2)
    """风场的风速绝对值"""

    # shape :(25,732)
    df_ws_forecast = pd.DataFrame(df_ws_forecast)

    print(f'df_ws_forecast shape:{df_ws_forecast.shape}')
    print(f'df_ws_subset shape:{df_ws_subset.shape}')
    """
        df_ws_forecast shape:(25, 732)
        df_ws_subset shape:(25, 732)
    """

    # TODO:[*] 25-05-11 由于数据可能存在空置（大概40个预报时次，20天没有预报数据)
    # 删除 df_ws_forecast 中的空值，并获取有效索引
    # axis=1 按列检索，若列中存在nan则删除该列
    valid_columns = df_ws_forecast.dropna(axis=1).columns

    # 使用有效索引筛选 df_ws_forecast 和 df_ws_subset
    df_ws_forecast_cleaned = df_ws_forecast[valid_columns]
    df_ws_subset_cleaned = df_ws_subset[valid_columns]
    print(f'df_ws_forecast_cleaned shape:{df_ws_forecast_cleaned.shape}')
    print(f'df_ws_subset_cleaned shape:{df_ws_subset_cleaned.shape}')
    print(f'df_ws_forecast_cleaned中包含 NaN 的行数:{df_ws_forecast_cleaned.isnull().any(axis=0).sum()}')
    print(f'df_ws_subset_cleaned NaN 的行数:{df_ws_subset_cleaned.isnull().any(axis=0).sum()}')
    print('# ————————————————————————————————————————————')
    # ————————————————————————————————————————————

    # 使用线性插值填充 NaN 值
    df_interpolated = df_ws_subset_cleaned.interpolate(method='linear')
    print(f'df_interpolated中包含 NaN 的行数:{df_interpolated.isnull().any(axis=0).sum()}')
    print('# ————————————————————————————————————————————')
    # 由于实况数据中也可能包含nan，需要找到实况数据中的nan的行，并双向删除
    # TODO:[-] 25-05-11
    # axis=0 ，表示按行进行操作 ; axis=1，表示按列进行操作 .
    valid_realdata_columns = df_interpolated.dropna(axis=1).columns
    df_ws_forecast_cleaned = df_ws_forecast[valid_realdata_columns]
    df_ws_subset_cleaned = df_interpolated[valid_realdata_columns]
    print(f'df_ws_forecast_cleaned shape:{df_ws_forecast_cleaned.shape}')
    print(f'df_ws_subset_cleaned shape:{df_ws_subset_cleaned.shape}')
    print(f'df_ws_forecast_cleaned中包含 NaN 的行数:{df_ws_forecast_cleaned.isnull().any(axis=0).sum()}')
    print(f'df_ws_subset_cleaned NaN 的行数:{df_ws_subset_cleaned.isnull().any(axis=0).sum()}')
    print('# ————————————————————————————————————————————')
    """
        df_ws_forecast_cleaned shape:(25, 612)
        df_ws_subset_cleaned shape:(25, 612)
    """
    # ————————————————————————————————————————————

    # step2: 数据格式化
    # 将 DataFrame 转换为 NumPy 数组
    X = []
    y = []

    # 再次检查清理后的df是否还存在空值
    # 检查 df_ws_forecast 是否有缺失值
    # cleaned_forecast = df_ws_forecast_cleaned.isnull().sum()
    # cleaned_realdata = df_ws_subset_cleaned.isnull().sum()
    # print("Missing values in df_ws_forecast_cleaned:\n", cleaned_forecast)
    # print("Missing values in df_ws_subset_cleaned:\n", cleaned_realdata)

    rows: int = df_ws_forecast_cleaned.shape[0]
    cols: int = df_ws_forecast_cleaned.shape[1]

    X_data_raw = df_ws_forecast_cleaned.values.T
    y_data_raw = df_ws_subset_cleaned.values.T

    # 如果尚未完成，请确保正确的重塑 (样本数, 时间步数)
    # 假设 df.values.T 得到 (样本数, 时间步数)
    X = X_data_raw.reshape(cols, rows, 1)
    y = y_data_raw.reshape(cols, rows, 1)

    if np.isnan(X).any() or np.isnan(y).any():
        print("数据中存在 NaN 值，请处理。")
        # 在继续之前处理NaN，例如通过插值或移除
        # 在此示例中，我们假设NaN已处理或将引发错误
        raise ValueError("数据中发现NaN值。请进行预处理。")

    print(f'X 初始形状: {X.shape}')
    print(f'y 初始形状: {y.shape}')

    # 根据数据校正 n_timesteps
    n_timesteps_data = X.shape[1]  # 根据您的形状，这应该是 25
    n_features_data = X.shape[2]  # 这应该是 1

    # --- 数据缩放 (归一化) ---
    # X 数据的缩放器
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    # 将 X 重塑为2D以适应缩放器: (样本数 * 时间步数, 特征数)
    X_reshaped = X.reshape(-1, n_features_data)
    scaler_X.fit(X_reshaped)  # 如果X是纯输入，为了保持一致性，在所有X数据上拟合
    # 注意：严格来说，对于训练/测试集划分，仅在X的训练部分拟合。
    # 然而，如果X是一个外部预测，其尺度独立于y的尺度，
    # 那么在所有X上拟合是可以的。对于从训练数据派生的特征，则仅在训练集上拟合。
    # 为简单起见，这里假设X是一个独立的输入。
    X_scaled_reshaped = scaler_X.transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(X.shape)

    # y 数据 (目标值) 的缩放器
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    # 将 y 重塑为2D以适应缩放器: (样本数 * 时间步数, 特征数)
    y_reshaped = y.reshape(-1, n_features_data)
    # 我们将在划分数据后，仅在 y 的训练部分拟合 scaler_y
    # y_scaled_reshaped = scaler_y.transform(y_reshaped) # 先不要转换所有的 y
    # y_scaled = y_scaled_reshaped.reshape(y.shape) # 先不要转换所有的 y

    # --- 数据划分 ---
    # 重要提示：对于时间序列，请使用 shuffle=False 或手动切片。
    # 使用 shuffle=True (默认值) 对于时序数据通常是不正确的。
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=False  # 将 shuffle 改为 False
    )

    # 在 y_train_orig 上拟合 scaler_y，并转换 y_train_orig 和 y_test_orig
    y_train_reshaped_for_scaler = y_train_orig.reshape(-1, n_features_data)
    scaler_y.fit(y_train_reshaped_for_scaler)

    y_train_scaled_reshaped = scaler_y.transform(y_train_reshaped_for_scaler)
    y_train_scaled = y_train_scaled_reshaped.reshape(y_train_orig.shape)

    y_test_scaled_reshaped = scaler_y.transform(y_test_orig.reshape(-1, n_features_data))
    y_test_scaled = y_test_scaled_reshaped.reshape(y_test_orig.shape)

    # 如果模型输出 (None, 25)，则调整 y_train_scaled 和 y_test_scaled
    # 模型的训练目标应与其输出形状匹配。
    # 如果模型输出 (None, n_timesteps_data)，则 y 需要是 (None, n_timesteps_data)
    y_train_fit = y_train_scaled.squeeze(axis=-1)
    y_test_fit = y_test_scaled.squeeze(axis=-1)

    print(f'X_train 形状: {X_train.shape}')
    print(f'X_test 形状: {X_test.shape}')
    print(f'y_train_fit 形状 (用于模型): {y_train_fit.shape}')
    print(f'y_test_fit 形状 (用于模型): {y_test_fit.shape}')
    print(f'y_train_orig 形状: {y_train_orig.shape}')  # 原始y_train，用于反归一化参考
    print(f'y_test_orig 形状: {y_test_orig.shape}')  # 原始y_test，用于最终MSE计算

    # --- 创建 LSTM 模型 ---
    # 模型调整为输出 (None, n_timesteps_data) 例如 (None, 25)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128, return_sequences=True,
                                 activation='relu',  # tanh 也很常用
                                 input_shape=(n_timesteps_data, n_features_data))))
    model.add(Dropout(0.2))
    # 第二个 BiLSTM 设置 return_sequences=False 以输出 (None, units*2)
    model.add(Bidirectional(LSTM(units=64, return_sequences=False, activation='relu')))
    model.add(Dropout(0.2))
    # Dense 层得到 (None, n_timesteps_data) 的输出
    model.add(Dense(n_timesteps_data))  # 输出 25 个值

    # 编译模型
    model.compile(optimizer='adam', loss=rmse)  # 使用您自定义的/Keras的rmse
    model.summary()

    # 训练模型
    history = model.fit(X_train, y_train_fit,
                        epochs=5,  # 测试时可考虑减少轮数，或使用 EarlyStopping
                        batch_size=16,
                        validation_data=(X_test, y_test_fit),
                        verbose=1)

    # --- 模型预测 ---
    predictions_scaled = model.predict(X_test)
    print("predictions_scaled (缩放后的预测) 形状:", predictions_scaled.shape)  # 应该是 (147, 25)

    # --- 对预测结果进行反归一化 ---
    # predictions_scaled 是 (测试样本数, 时间步数) 例如 (147, 25)
    # scaler_y 是在形状为 (-1, 特征数) 例如 (-1, 1) 的数据上拟合的
    # 我们需要重塑 predictions_scaled 以便与 inverse_transform 兼容
    predictions_scaled_for_inverse = predictions_scaled.reshape(-1, n_features_data)
    predictions_inverted_flat = scaler_y.inverse_transform(predictions_scaled_for_inverse)
    # 重塑回 (测试样本数, 时间步数)
    predictions_final = predictions_inverted_flat.reshape(predictions_scaled.shape)
    print("predictions_final (反归一化后的预测) 形状:", predictions_final.shape)

    # --- 在原始尺度上用 MSE 评估模型 ---
    # y_test_orig 是 (测试样本数, 时间步数, 特征数)，例如 (147, 25, 1)
    # predictions_final 是 (测试样本数, 时间步数)，例如 (147, 25)
    # 我们需要公平地比较它们。压缩 y_test_orig 的最后一个维度。
    y_test_orig_squeezed = y_test_orig.squeeze(axis=-1)

    # 计算所有预测点的 MSE
    mse = mean_squared_error(y_test_orig_squeezed.flatten(), predictions_final.flatten())
    print(f'测试集 MSE (在原始尺度上): {mse}')

    # model.evaluate 报告的损失是在缩放数据上使用 'rmse' 指标计算的
    loss_on_scaled_data = model.evaluate(X_test, y_test_fit, verbose=0)
    print(f'测试集损失 (在缩放数据上的 RMSE，与 model.compile 一致): {loss_on_scaled_data}')

    # 如果需要，计算每个样本的 MSE
    # mse_per_sample = np.mean((y_test_orig_squeezed - predictions_final)**2, axis=1)
    # print(f'每个样本的平均 MSE (在原始尺度上): {np.mean(mse_per_sample)}')


if __name__ == '__main__':
    main()
