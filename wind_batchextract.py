import csv

import arrow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import pathlib
import xml.etree.ElementTree as ET
import xarray as xr
import codecs
import datetime


# 先从海浪数据中提取出经纬度，时间，风，海浪高度
# 解析单个文件，并存于字典内
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


def traning():
    # 假设我们有以下输入数据
    # u_data 和 v_data 是形状为 (730, 72, 3) 的风场数据
    # ybg_data 是形状为 (730, 3) 的浮标数据
    u_data = np.random.rand(730, 72, 3)  # 730个发布时次
    v_data = np.random.rand(730, 72, 3)
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


def main():
    read_path: str = r'Z:\WIND\GRAPES\2024'
    out_put_path: str = r'./data'
    out_put_file_path: str = str(pathlib.Path(out_put_path) / 'GRAPES_2024_24')
    lat: float = 39.5003
    lng: float = 120.59533
    # 暂时注释掉批量生成file
    # for index in range(11):
    #     temp_month = index + 1
    #     df = batch_readncfiles(read_path, lat, lng, temp_month)
    #     df.to_csv(f'out_put_file_path_{temp_month}.csv')
    # 生成测试数据
    simulate_forcast_data()
    pass


if __name__ == '__main__':
    main()
