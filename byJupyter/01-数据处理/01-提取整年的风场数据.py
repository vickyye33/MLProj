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

from utils.common import merge_dataframe


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


def batch_readncfiles_byyears(read_path: str, out_put_path: str, lat: float, lng: float, year: int = 2024,
                              month: Optional[int] = None, count: int = 60):
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
    # out_put_path: str = r'G:\05DATA\01TRAINING_DATA\WIND'
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
                                u_vals = filter_ds['UGRD_10maboveground'].values[:count]
                                v_vals = filter_ds['VGRD_10maboveground'].values[:count]
                                # TODO:[-] 25-05-21 若数据有空缺，对数据进行填充
                                u_vals_padded = np.pad(u_vals, (0, count - len(u_vals)), constant_values=np.nan)
                                v_vals_padded = np.pad(v_vals, (0, count - len(v_vals)), constant_values=np.nan)

                                dt_vals = filter_ds['time'].values
                                dt64_forecast_start = dt_vals[0]
                                dt_forecast_start: datetime = pd.to_datetime(dt64_forecast_start)
                                dt_forecast_start_str: str = dt_forecast_start.strftime('%Y%m%d%H%M%S')
                                temp_u_column_name: str = f'{dt_forecast_start_str}_u'
                                temp_v_column_name: str = f'{dt_forecast_start_str}_v'
                                df_dict[temp_u_column_name] = u_vals_padded
                                df_dict[temp_v_column_name] = v_vals_padded
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


def main():
    read_path: str = r'/Volumes/DATA/WIND/GRAPES/2024'
    # read_path: str = r'/Volumes/DATA/WIND/TEST/2024'
    out_put_path: str = r'/Volumes/DATA/01TRAINNING_DATA/WIND/01'
    out_put_file_path: str = str(pathlib.Path(out_put_path) / 'GRAPES_2024_24')
    lat: float = 39.5003
    lng: float = 120.59533
    # step1:批量读取nc文件并提取指定经纬度的72小时数据并拼接成 dataframe
    # for index in np.arange(1, 12):
    #     temp_month = index + 1
    #     batch_readncfiles_byyears(read_path, out_put_path, lat, lng, 2024, temp_month, 72)

    # step2:将上面step1处理的按月保存的72小时浮标站位的时序数据合并为一整年的数据
    merge_df = merge_dataframe(out_put_path)
    merge_full_path: str = str(pathlib.Path(out_put_path) / 'merge.csv')
    merge_df.to_csv(merge_full_path)

    # -------------

    pass


if __name__ == '__main__':
    main()
