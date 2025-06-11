import csv
from typing import Optional

import arrow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src.layers import LSTM, Dropout, Dense
from pandas import DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pathlib
import xml.etree.ElementTree as ET
import xarray as xr
import codecs
import datetime

# 先从海浪数据中提取出经纬度，时间，风，海浪高度
# 解析单个文件，并存于字典内
from sklearn.preprocessing import StandardScaler


def batch_readxmlfiles(read_file: str):
    parser = ET.XMLParser(encoding="iso-8859-5")
    # print(read_file)
    Tree = ET.parse(read_file, parser=parser)
    header = []
    row = []
    root = Tree.getroot()
    # print(root.tag)
    dict_temp = {}
    time_node = root.find('./BuoyageRpt/DateTime')
    # print(time_node.tag)
    dict_temp["time"] = time_node.get("DT")
    location_node = root.find('./BuoyageRpt/BuoyInfo/Location')
    longitude_temp = location_node.get("longitude").replace("Ёф", "").replace("E", "")
    min_temp = longitude_temp.split("Ёу")
    longitude = float(min_temp[0]) + float(min_temp[1]) / 60
    dict_temp["longitude"] = longitude
    latitude_temp = location_node.get("latitude").replace("Ёф", "").replace("N", "")
    min_temp = latitude_temp.split("Ёу")
    latitude = float(min_temp[0]) + float(min_temp[1]) / 60
    dict_temp["latitude"] = latitude
    # print(dict_temp["longitude"])
    # print(dict_temp["latitude"])
    BD_node = root.find('./BuoyageRpt/HugeBuoyData/BuoyData')
    dict_temp["WS"] = BD_node.get("WS")
    dict_temp["YBG"] = BD_node.get("YBG")
    return dict_temp


def batch_readncfiles(read_path: str, lat: float, lng: float, month: int = 1):
    """
        根据指定路径遍历该路径下的所有文件，并读取每个文件中的[0,23]h的时序数据(根据经纬度)
    :param read_path: 读取nc的根目录
    :param lat:
    :param lng:
    :return:
    """

    df_nc: xr.Dataset = None
    nc_path = pathlib.Path(read_path)
    df_dict: dict = {}
    for file in nc_path.iterdir():
        # for file in 'GRAPES_2024010100_240h_UV.nc','GRAPES_2024010112_240h_UV.nc':
        # 修改为按月切分
        if file.is_file():
            # 获取当前文件的 datetime 字符串
            dt_str: str = file.name.split('_')[1]
            dt_arrow = arrow.get(dt_str, 'YYYYMMDDHH')
            temp_month = dt_arrow.month
            if month == temp_month:
                # step1: 拼接成文件全路径
                # file_full_path = nc_path / file
                file_full_path_str: str = str(file)
                # step2: 使用 xarray.open_dataset 打开 netcdf文件
                temp_df: xr.Dataset = xr.open_dataset(file_full_path_str)
                # 注意: 打开的 Dataset 有三个维度,目前只需要按照经纬度提取几个位置的24h内的时序数据
                if temp_df is not None:
                    """
                        Coordinates:
                        * latitude             (latitude) float64 -89.94 -89.81 -89.69 ... 89.81 89.94
                        * longitude            (longitude) float64 0.0 0.125 0.25 ... 359.8 359.9
                        * time                 (time) datetime64[ns] 2024-01-01 ... 2024-01-11
                        Data variables:
                            UGRD_10maboveground  (time, latitude, longitude) float32 ...
                            VGRD_10maboveground  (time, latitude, longitude) float32 ...
                    """
                    # 从该文件中提取指定经纬度的时序数据
                    filter_ds = temp_df.sel(latitude=lat, longitude=lng, method='nearest')
                    # 分别取出 u 与 v 分量
                    u_vals = filter_ds['UGRD_10maboveground'].values[:25]
                    v_vals = filter_ds['VGRD_10maboveground'].values[:25]
                    dt_vals = filter_ds['time'].values
                    dt64_forecast_start = dt_vals[0]
                    dt_forecast_start: datetime = pd.to_datetime(dt64_forecast_start)
                    dt_forecast_start_str: str = dt_forecast_start.strftime('%Y%m%d%H%M%S')
                    temp_u_column_name: str = f'{dt_forecast_start_str}_u'
                    temp_v_column_name: str = f'{dt_forecast_start_str}_v'
                    df_dict[temp_u_column_name] = u_vals
                    df_dict[temp_v_column_name] = v_vals
                    print(f"读取{file_full_path_str}成功")
                else:
                    df_nc = temp_df
    # 将最终的 dict -> pd.DataFrame
    df = pd.DataFrame(df_dict)
    print('生成最终DataFrame ing ')
    return df


def batch_readncfiles_byyears(read_path: str, lat: float, lng: float, year: int = 2024, month: Optional[int] = None):
    """
        根据指定路径遍历该路径下的所有文件，并读取每个文件中的[0,23]h的时序数据(根据经纬度)
    :param read_path: 读取nc的根目录
    :param lat:
    :param lng:
    :return:
    """

    df_nc: xr.Dataset = None
    nc_path = pathlib.Path(read_path)
    df_dict: dict = {}
    out_put_path: str = r'G:\05DATA\01TRAINING_DATA\WIND'
    for file in nc_path.iterdir():
        # for file in 'GRAPES_2024010100_240h_UV.nc','GRAPES_2024010112_240h_UV.nc':
        # 修改为按月切分
        if file.is_file():
            try:
                # 获取当前文件的 datetime 字符串
                dt_str: str = file.name.split('_')[1]
                dt_arrow = arrow.get(dt_str, 'YYYYMMDDHH')
                temp_year = dt_arrow.year
                temp_month = dt_arrow.month
                if year == temp_year:
                    if month is not None:
                        if temp_month == month:
                            # step1: 拼接成文件全路径
                            # file_full_path = nc_path / file
                            file_full_path_str: str = str(file)
                            # step2: 使用 xarray.open_dataset 打开 netcdf文件
                            temp_df: xr.Dataset = xr.open_dataset(file_full_path_str)
                            # 注意: 打开的 Dataset 有三个维度,目前只需要按照经纬度提取几个位置的24h内的时序数据
                            if temp_df is not None:
                                """
                                    Coordinates:
                                    * latitude             (latitude) float64 -89.94 -89.81 -89.69 ... 89.81 89.94
                                    * longitude            (longitude) float64 0.0 0.125 0.25 ... 359.8 359.9
                                    * time                 (time) datetime64[ns] 2024-01-01 ... 2024-01-11
                                    Data variables:
                                        UGRD_10maboveground  (time, latitude, longitude) float32 ...
                                        VGRD_10maboveground  (time, latitude, longitude) float32 ...
                                """
                                # 从该文件中提取指定经纬度的时序数据
                                filter_ds = temp_df.sel(latitude=lat, longitude=lng, method='nearest')
                                # 分别取出 u 与 v 分量
                                u_vals = filter_ds['UGRD_10maboveground'].values[:25]
                                v_vals = filter_ds['VGRD_10maboveground'].values[:25]
                                dt_vals = filter_ds['time'].values
                                dt64_forecast_start = dt_vals[0]
                                dt_forecast_start: datetime = pd.to_datetime(dt64_forecast_start)
                                dt_forecast_start_str: str = dt_forecast_start.strftime('%Y%m%d%H%M%S')
                                temp_u_column_name: str = f'{dt_forecast_start_str}_u'
                                temp_v_column_name: str = f'{dt_forecast_start_str}_v'
                                df_dict[temp_u_column_name] = u_vals
                                df_dict[temp_v_column_name] = v_vals
                                print(f"读取{file_full_path_str}成功")
                            else:
                                df_nc = temp_df
            except Exception as e:
                print(e.args)
    # 将最终的 dict -> pd.DataFrame
    df = pd.DataFrame(df_dict)
    print('生成最终DataFrame ing ')
    out_put_filename: str = f'out_put_MF01001_{str(year)}_{str(month)}.csv'
    save_path: str = str(pathlib.Path(out_put_path) / out_put_filename)
    df.to_csv(save_path)
    print(f'存储路径:{save_path}')
    # return df


def simulate_forcast_data():
    """
        测试生成三个浮标数据
    :return:
    """
    # (730, 72, 3)
    # 每个发布时刻一个 72 *3 的集合
    u_data = np.random.rand(365 * 2, 72, 3)
    v_data = np.random.rand(365 * 2, 72, 3)
    ybg_data = np.random.rand(730, 3)
    pass


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
        start_time = '2024-01-01 00:00:00'
        end_time = '2024-12-31 23:00:00'
        time_series = pd.date_range(start=start_time, end=end_time, freq='H')

        # 将time列的内容从int64 => str
        df['time'] = df['time'].astype(str)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
        # step2: 将 time列设置为index，并将index替换为标准时间集合
        df.set_index('time', inplace=True)
        df_reindexed = df.reindex(time_series)
        df_reindexed.index.name = 'time'

        # step3: 生成12小时为间隔的时间数组
        freq_str: str = f'{issue_hours_steps}H'
        issue_dt_series = pd.date_range(start=start_time, end=end_time, freq=freq_str)

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


def traning_ws(test_read_path: str, training_read_path: str):
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
                    u_data_dict[dt_temp_str] = col_vector.tolist()
                elif var_temp_str == 'v':
                    v_data_dict[dt_temp_str] = col_vector.tolist()
                print(f'当前列:{col_name}处理成功~')
            except Exception as e:
                print(f'当前列:{col_name}处理错误!')
        # step2: 将字典统一转换为二维数组
        result_u_array = [val for key, val in u_data_dict.items()]
        result_v_array = [val for key, val in v_data_dict.items()]
        pass


def traning():
    # 假设我们有以下输入数据
    # u_data 和 v_data 是形状为 (730, 72, 3) 的风场数据
    # ybg_data 是形状为 (730, 3) 的浮标数据
    u_data = np.random.rand(730, 72, 3)  # 730个发布时次
    v_data = np.random.rand(730, 72, 3)
    # 此处应该修改为 365*2 个发布时间，对应的未来72小市场的实际观测值，3个坐标位置。
    # 形状修改为 (730,72,3)
    ybg_data = np.random.rand(730, 3)

    # 将u和v数据合并为输入特征
    features = np.concatenate((u_data, v_data), axis=-1)  # 形状为 (730, 72, 6)

    # 将输入特征和输出标签划分为训练集和测试集
    """
        features:
            这是模型的输入数据，通常是一个二维数组或数据框（DataFrame），每一行代表一个样本，每一列代表一个特征。模型将使用这些特征来进行学习和预测。
        ybg_data:
            这是模型的输出或目标标签数据，通常是一维数组或数据框（DataFrame），每个元素对应于features中的一个样本的标签。模型将学习如何从特征中预测这些标签。
    """
    X_train, X_test, y_train, y_test = train_test_split(features, ybg_data, test_size=0.2, random_state=42)

    # 构建神经网络模型
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(72, 6)),  # 展平输入
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)  # 输出层，预测三个经纬度位置的ybg值
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

    # 评估模型
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    # 预测未来72小时的ybg值
    future_u_data = np.random.rand(1, 72, 3)  # 示例未来风场数据
    future_v_data = np.random.rand(1, 72, 3)
    future_features = np.concatenate((future_u_data, future_v_data), axis=-1)
    ybg_prediction = model.predict(future_features)
    print(f"Predicted YBG values: {ybg_prediction}")


def train_data_preparation(u_data: pd.DataFrame, v_data: pd.DataFrame, ws_data: pd.DataFrame, time_step=24):
    # 合并 u 和 v 数据
    # u_data :(25,613),
    # ws_data:(72,732)
    # TODO:[*] 25-04-29 从以上数据可知u 与 ws 的数据集形状不同，需要标准化
    # ws是正常的 366*2=732
    # u 是有缺失的
    # shape:(25,1226)
    # shape:(25, 1464) u|v 单独为 (25,732)
    # 注意 (n,m) 表示的是 n 行，m列，其中 n 表示样本的数量, m表示特征的数量
    # (25,732) 即表示 25 行，732 列，需要将 732 表示特征，25 表示样本
    # TODO:[-] 25-05-07 将 u , v => (25,732,2)
    # features = pd.concat([u_data, v_data], axis=1)
    features = np.stack((u_data, v_data), axis=-1)
    # 合并和可通过 features_scaled[...,0] 查看 u_data

    # TODO:[*] 25-04-29 此处的意义是什么?
    """
        StandardScaler 是 scikit-learn 库中的一个类，用于标准化特征。它通过将每个特征的值减去该特征的均值，然后除以该特征的标准差来实现标准化。标准化后的特征将具有均值为 0 和标准差为 1。
        features 是一个形状为 (25, 1464) 的数组，表示有 25 个样本和 1464 个特征。
        features_scaled 将包含标准化后的数据，形状仍然是 (25, 1464)，但每个特征的分布将变为均值为 0 和标准差为 1。
    """
    # 由于目前为三维矩阵 (25,732,2) 需要转换为二维数组再使用 StandardScaler 进行归一化
    features_reshaped = features.reshape(-1, 2)
    # 标准化
    scaler = StandardScaler()
    # shape:(25, 1464)
    features_scaled_reshaped = scaler.fit_transform(features_reshaped)
    """归一化后的合并后(features)的特征值数据集"""

    features_scaled = features_scaled_reshaped.reshape(25, 732, 2)

    # 创建时间序列数据

    def create_dataset(features, targets, time_step=1):
        """
            TODO:[*] 25-05-07 此处修改为按照 (25,732,2) shape 进行创建数据集操作
            X 的shape：每个样本的形状为 (time_step, 732, 2)，其中 time_step 可以根据您的需求进行调整。
            y 的shape：每个目标值与 X 中的样本对应，通常是单一值（例如，预测的下一个时间步的值）。
        :param features:
        :param targets:
        :param time_step:
        :return:
        """
        X, y = [], []
        # len(features) = 25 ，相当于是行数
        # 此处的 time_step 应修改为1，时间不长为1hour？
        # TODO:[*] 25-05-05 ERROR:
        if len(features) <= time_step:
            raise ValueError("features 的长度必须大于 time_step")
        for i in range(len(features) - time_step):
            # 若 ts=24的话, features[0:24, :]
            # shape (24, 1464)
            X.append(features[i:(i + time_step), :])
            # 若 ts=24的话，targets[0+24]
            y.append(targets[i + time_step])
        return np.array(X), np.array(y)

    # TODO:[*] 25-05-05 此处的时间步长是否有问题？
    # time_step = 24  # 例如，使用过去25小时的数据

    """
        X 是一个三维数组，形状为 (样本数, 时间步, 特征数)，其中：
        样本数：表示用于训练模型的时间序列样本的数量。
        时间步：表示每个样本中包含的历史时间点的数量（在本例中为 25，表示过去 25 小时的风场数据）。
        特征数：表示每个时间步的特征数量。在你的情况下，u_data 和 v_data 合并后，每个时间步的特征数量为 2（即每个时间点的 u 和 v 分量）。
        示例
        如果 u_data 和 v_data 各有 100 个时间点，且我们设置时间步为 25，则：        
        X 的形状为 (76, 25, 2)，因为每个样本需要 25 个时间步的数据，最后一个样本可以从第 76 个时间点开始。
    """

    """
        y 是一个一维数组，形状为 (样本数,)，表示每个样本对应的目标值。在这个案例中，y 包含的是对应时间步之后 25 小时的风速（ws）观测值。
        示例
        继续以上的例子，y 将包含从第 26 到第 100 个时间点的 ws 值，因此：        
        y 的形状为 (76,)，与 X 的样本数相对应。
    """
    # TODO:[*] 25-04-29 X 与 y的 shape均为0
    # X shape:(1,24,1464)
    # y shape:(1,732)
    # y 若为 1,732 说明是 一年 366*2 两个月报时次，是不对的
    X, y = create_dataset(features_scaled, ws_data.values, time_step)
    return X, y


def train_instruct_model(X, y):
    """
        构建 LSTM 模型
    :param X:
    :param y: 暂未使用
    :return:
    """
    # step1:创建一个顺序模型，这是 Keras 中构建神经网络的一种方式，适合逐层添加网络结构。
    model = Sequential()
    # step2:添加LSTM层，50个单元
    # return_sequences=True 表示该层将返回序列数据，以便可以堆叠更多的 LSTM 层。
    # input_shape 指定输入数据的形状，通常为 (时间步数, 特征数量)。
    # IndexError: tuple index out of range
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    # 添加一个 Dropout 层，以防止过拟合。这里的 0.2 表示在训练过程中随机丢弃 20% 的神经元。
    model.add(Dropout(0.2))
    # 添加另一个 LSTM 层，返回序列的特性不再需要，因此不设置 return_sequences。
    model.add(LSTM(units=50))

    model.add(Dropout(0.2))
    # 添加一个全连接层（Dense），输出一个值，适用于回归任务。
    model.add(Dense(units=1))  # 输出层
    # 编译模型
    # 使用 Adam 优化器，这是一个常用的优化算法。
    # 设置损失函数为均方误差（mean squared error），适合回归问题。
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_fit_model(model, X, y):
    # 拆分数据集为训练集和测试集
    train_size = int(len(X) * 0.8)
    # 根据计算出的 train_size 将 X 拆分为训练集和测试集。
    X_train, X_test = X[:train_size], X[train_size:]
    # 将目标变量 y 拆分为训练集和测试集。
    y_train, y_test = y[:train_size], y[train_size:]
    # 训练模型
    """
        使用 fit 方法训练模型。
        epochs=100：训练 100 个周期（epochs）。
        batch_size=32：每个批次的样本数量为 32。
        validation_data=(X_test, y_test)：在每个周期结束时使用测试集数据进行验证，这样可以监控模型的性能并防止过拟合。
        X_train 训练数据的输入特征，通常是一个多维数组或矩阵。每一行代表一个样本，每一列代表一个特征。
        y_train y_train 是训练数据的目标值，通常是一个一维数组或向量。每个元素对应 X_train 中的样本，表示该样本的期望输出。
        TODO:[-] 25-05-07
        
    """
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    # 返回训练后的模型，以及测试集的特征和目标变量。这可以用于后续的评估或预测。
    return model, X_test, y_test


def train_model_evaluate(model, X_test, y_test):
    # 进行预测
    predictions = model.predict(X_test)

    # 反标准化（如果需要）
    # predictions = scaler.inverse_transform(predictions)

    # TODO:[*] 25-05-06 此处出错
    """
          ValueError: Input contains NaN.
            python-BaseException
    """
    # 评估模型
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')


def merge_dataframe(read_path: str) -> Optional[pd.DataFrame]:
    """
        读取指定路径，遍历该路径下的所有文件，将读取的dataframe横向拼接
    :param read_path:
    :return:
    """
    first_df: Optional[pd.DataFrame] = None
    if pathlib.Path(read_path).exists():
        for file in pathlib.Path(read_path).iterdir():
            temp_df: pd.DataFrame = pd.read_csv(str(file))
            if first_df is None:
                first_df = temp_df
            else:
                first_df = pd.concat([first_df, temp_df], axis=1)
    return first_df


def main():
    read_path: str = r'Z:\WIND\GRAPES\2024'
    out_put_path: str = r'./data'
    out_put_file_path: str = str(pathlib.Path(out_put_path) / 'GRAPES_2024_24')
    lat: float = 39.5003
    lng: float = 120.59533
    """
        TODO:[*] 25-04-24 ERROR:
            KeyError: [<function _open_scipy_netcdf at 0x00000190473A7790>, ('Z:\\WIND\\GRAPES\\2024\\GRAPES_2024050100_240h_UV.nc',), 'r', (('mmap', None), ('version', 2)), '6432fc26-d2f4-4ff2-8c72-c83c40ef2b3a']
            During handling of the above exception, another exception occurred:
            ——————————————————————————
            ValueError: When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.
            python-BaseException
    """
    # step1:批量读取nc文件并提取指定经纬度的72小时数据并拼接成 dataframe
    # for index in range(11):
    #     temp_month = index + 1
    #     batch_readncfiles_byyears(read_path, lat, lng, 2024, temp_month)

    # step2:将上面step1处理的按月保存的72小时浮标站位的时序数据合并为一整年的数据
    # merge_df = merge_dataframe(r'G:\05DATA\01TRAINING_DATA\WIND')
    # merge_df.to_csv(r'G:\05DATA\01TRAINING_DATA\WIND\merge.csv')

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

    # TODO:[-] 25-05-05 注意: 此处通过 df_u.loc[:,0]的方式取出第一列;方式2：取出第一列对应的列名 df_u.columns[0]
    # eg 第一列: 2024-01-01 00:00:00
    # 取出第一列的列向量

    # step2: 只提取 ws
    # df_ws.to_csv(out_put_file_path)
    # TODO:[*] 25-05-05
    # 此处将 时间步长修改为 1h ，输出的 X shape:(24,1,1464),y shape:(24,732)
    #
    X, y = train_data_preparation(df_u, df_v, df_ws, 24)
    model = train_instruct_model(X, y)
    model, X_test, y_test = train_fit_model(model, X, y)
    train_model_evaluate(model, X_test, y_test)

    # TODO:[-] 25-04-28 以上部分分别处理并获取了 test 的 u & v dataframe ; 以及 ws 的 dataframe

    # 暂时注释掉批量生成file
    # for index in range(11):
    #     temp_month = index + 1
    #     df = batch_readncfiles(read_path, lat, lng, temp_month)
    #     df.to_csv(f'out_put_file_path_{temp_month}.csv')
    # 生成测试数据
    # simulate_forcast_data()
    pass


if __name__ == '__main__':
    main()
