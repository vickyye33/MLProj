import csv
from typing import Optional

import arrow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src.layers import LSTM, Dropout, Dense, BatchNormalization
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
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


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


def model_fit():
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
    # ----------------------
    # TODO:[-] 25-05-18 根据阈值进行过滤，超出部分剔除
    min = 0
    max = 30
    df_ws_forecast = df_ws_forecast.where((df_ws_forecast > min) & (df_ws_forecast < max))
    df_ws_subset = df_ws_subset.where((df_ws_subset > min) & (df_ws_subset < max))
    print(f'过滤后的预测 df shape:{df_ws_forecast.shape}')
    print(f'过滤后的实况 df shape:{df_ws_subset.shape}')
    # -----------------------
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
    # -----------------------
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
    # -------------------------------
    # 获取 X 的维度信息
    n_samples, n_timesteps, n_features = X.shape  # n_samples=608, n_timesteps=25, n_features=1

    # 对 X 数据进行归一化处理
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    # 由于 MinMaxScaler 只能处理二维数组，因此需要先将 X 转换为二维，再归一化，最后再恢复到三维结构。
    X_2d = X.reshape(-1, n_features)  # 转换为 (608*25, 1) 的二维数组
    X_scaled_2d = scaler_X.fit_transform(X_2d)  # 进行归一化
    X_norm = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)  # 恢复为 (608, 25, 1)
    # 对 y 数据进行归一化处理
    # 由于 y 中可能含有 NaN 值，这里先将 NaN 替换为 0（或其它合适的值），再进行归一化
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_2d = y.reshape(-1, n_features)
    y_2d_no_nan = np.nan_to_num(y_2d, nan=0.0)  # 将 NaN 替换为 0.0
    y_scaled_2d = scaler_y.fit_transform(y_2d_no_nan)
    y_norm = y_scaled_2d.reshape(n_samples, n_timesteps, n_features)
    # ----------------

    # 由于模型输出的 Dense 层设置为输出25个节点，
    # 我们将 y 的形状从 (608, 25, 1) 调整为 (608, 25)
    y_norm = y_norm[:, :, 0]

    # 3. 数据集划分 (80% 训练集，20% 测试集)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

    # -----------------
    model = Sequential()
    # 添加 LSTM 层：
    #   - 50 个隐藏单元
    #   - 激活函数使用 'relu'
    #   - 输入形状为 (时间步长, 特征数) 即 (25, 1)
    # 第一层 LSTM 层，return_sequences=True 表示返回整个序列，为后续 LSTM 层提供输入
    model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
    # 加入 Dropout 层防止过拟合（可选）
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # 第二层 LSTM 层，通过减小神经元数量形成特征压缩，同时保持时序信息
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # 第三层 LSTM 层，此处不返回完整的序列，以便后接 Dense 层处理
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # 添加全连接层，用于进一步捕捉数据中的非线性关系
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))

    # 添加全连接(Dense)层，输出 25 个节点，对应预测 25 个时间步的数据
    model.add(Dense(n_timesteps))

    # ------------------------
    # 5. 编译模型：采用 adam 优化器和均方误差损失函数
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 使用 EarlyStopping 回调函数，在验证集上损失不再降低时提前停止训练
    # 设置回调：EarlyStopping 与 ReduceLROnPlateau
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    # v3: + min_lr=1e-05, factor 0.5 -> 0,7
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-05, verbose=1)
    # ------------------------
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=16,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop, reduce_lr])

    # -------------------------
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", test_loss)
    # -------------------------
    y_pred = model.predict(X_test)
    # -------------------------
    # 9. 逆归一化：将预测值和测试集的实际值转换回原始量纲
    # 预测值的逆处理
    y_pred_2d = y_pred.reshape(-1, 1)
    y_pred_inverse = scaler_y.inverse_transform(y_pred_2d)
    y_pred_inverse = y_pred_inverse.reshape(y_pred.shape)

    # 测试集实际值的逆处理
    y_test_2d = y_test.reshape(-1, 1)
    y_test_inverse = scaler_y.inverse_transform(y_test_2d)
    y_test_inverse = y_test_inverse.reshape(y_test.shape)

    # ------------------------
    # 注意：这里将多步预测（25 时间步）的结果全部展开后计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_test_inverse.reshape(-1), y_pred_inverse.reshape(-1)))
    print("RMSE:", rmse)
    # V1
    # RMSE: 2.451076278495527
    # V2
    # RMSE:3.7576
    # V3
    """
        Epoch 100/100
        31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 36ms/step - loss: 0.0024 - val_loss: 0.0013 - learning_rate: 6.1035e-08
        Test Loss: 0.0013040007324889302
        4/4 ━━━━━━━━━━━━━━━━━━━━ 2s 328ms/step
        RMSE: 3.7302610595411854
    """
    # V4
    """
        Epoch 91/100
        31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 37ms/step - loss: 0.0019 - val_loss: 0.0014 - learning_rate: 1.0000e-05
        Test Loss: 0.0013534461613744497
        4/4 ━━━━━━━━━━━━━━━━━━━━ 2s 389ms/step
        RMSE: 3.8003256457481336
    """
    # V5—— 过滤了异常值
    """
        Epoch 44: ReduceLROnPlateau reducing learning rate to 0.00024009999469853935.
        31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 38ms/step - loss: 0.0160 - val_loss: 0.0171 - learning_rate: 3.4300e-04
        Test Loss: 0.013991340063512325
        4/4 ━━━━━━━━━━━━━━━━━━━━ 3s 388ms/step
        RMSE: 2.7442119501128412
    """


def main():
    model_fit()


if __name__ == '__main__':
    main()
