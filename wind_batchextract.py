import csv
from typing import Optional

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
    merge_df = merge_dataframe(r'G:\05DATA\01TRAINING_DATA\WIND')
    merge_df.to_csv(r'G:\05DATA\01TRAINING_DATA\WIND\merge.csv')

    read_file_full_path: str = r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv'
    # step1: 生成一年的 365*2 =730 个 ws,ybg -> 只取 ws
    df = batch_get_realdata(read_file_full_path)
    # step2: 只提取 ws
    df.to_csv(out_put_file_path)

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
