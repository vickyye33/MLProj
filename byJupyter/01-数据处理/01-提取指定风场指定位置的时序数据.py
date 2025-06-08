import csv
from typing import Optional

import arrow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
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
                    # ec 的 u v 分量的 name
                    u_vals = filter_ds['u10'].values[:25]
                    v_vals = filter_ds['v10'].values[:25]
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


def batch_readncfiles_byyears(read_path: str, out_put_path, lat: float, lng: float, year: int = 2024,
                              month: Optional[int] = None):
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

    # name: eg : zhongyuan_ecmwf_det_atm_2024010100.nc

    for file in nc_path.iterdir():
        # for file in 'GRAPES_2024010100_240h_UV.nc','GRAPES_2024010112_240h_UV.nc':
        # 修改为按月切分
        if file.is_file():
            try:
                # 获取当前文件的 datetime 字符串
                # 对于GRAPES风场的时间戳
                dt_str: str = file.name.split('.')[0].split('_')[1]
                # 2024010100
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
                                u_vals = filter_ds['u10'].values[:25]
                                v_vals = filter_ds['v10'].values[:25]
                                dt_vals = filter_ds['time'].values
                                dt64_forecast_start = dt_vals[0]
                                dt_forecast_start: datetime = pd.to_datetime(dt64_forecast_start)
                                dt_forecast_start_str: str = dt_forecast_start.strftime('%Y%m%d%H%M%S')
                                temp_u_column_name: str = f'{dt_forecast_start_str}_u'
                                temp_v_column_name: str = f'{dt_forecast_start_str}_v'
                                # df_dict[temp_u_column_name] = u_vals
                                # df_dict[temp_v_column_name] = v_vals
                                df_dict[temp_u_column_name] = u_vals.reshape(-1)
                                df_dict[temp_v_column_name] = v_vals.reshape(-1)
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



def main():
    """
        不要使用此脚本
    :return:
    """
    # mac
    # read_path: str = r'/Volumes/DATA/WIND/ECMWF/2024'
    # read_path: str = r'/Users/evaseemefly/03data/02wind/2024'
    # out_put_path: str = r'/Users/evaseemefly/03data/02wind'
    # out_put_file_path: str = str(pathlib.Path(out_put_path) / 'GRAPES_2024_24')
    # razer
    read_path: str = r'Z:/WIND/GRAPES/2024'
    out_put_path: str = r'Z:/SOURCE_DATA'

    year = 2024
    month = 1
    lat: float = 39.5003
    lng: float = 120.59533

    # TODO:[*] 25-04-29 根据起止时间，时间步长为12h，生成发布时间index
    start_time = '2024-01-01 00:00:00'
    end_time = '2024-12-31 23:00:00'
    issue_times_index = pd.date_range(start=start_time, end=end_time, freq='12h')

    batch_readncfiles_byyears(read_path, out_put_path, lat, lng, year, month)

    pass


if __name__ == '__main__':
    main()
