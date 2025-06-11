import csv
from typing import Optional

import arrow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src.layers import LSTM, Dropout, Dense
from pandas import DatetimeIndex
from sklearn.impute import SimpleImputer
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
# from tensorflow.keras.callbacks import EarlyStopping
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
                # print(f'当前列:{col_name}处理成功~')
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
    # mac nas 地址
    test_read_path: str = r'/Volumes/DATA/01TRAINNING_DATA/WIND/merge.csv'
    train_read_path: str = r'/Volumes/DATA/FUB/MF01001/2024_local.csv'
    read_file_full_path: str = r'/Volumes/DATA/FUB/MF01001/2024_local.csv'
    # win nas 地址
    # read_file_full_path: str = r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv'
    # test_read_path: str = r'G:\05DATA\01TRAINING_DATA\WIND\merge.csv'
    # train_read_path: str = r'G:\05DATA\01TRAINING_DATA\FUB\MF01001\2024_local.csv'
    df_u, df_v = get_test_array(test_read_path,
                                train_read_path, issue_times_index)
    #  生成一年的 365*2 =730 个 ws,ybg -> 只取 ws
    # 注意实况 shape : (72,732)
    df_ws = batch_get_realdata(read_file_full_path)
    """shape:(72,732)"""
    df_ws_subset = df_ws.iloc[:25, :]
    min = 0
    max = 30
    df_ws_subset = df_ws_subset.where((df_ws_subset > min) & (df_ws_subset < max))
    df_ws_forecast = np.sqrt(df_u ** 2 + df_v ** 2)
    df_ws_forecast = df_ws_forecast.where((df_ws_forecast > min) & (df_ws_forecast < max))

    # 求 732 组 数据的 平均 RMSE
    rmse_list = []
    err_count = 0
    for i in range(732):
        try:
            y_pred = df_ws_forecast.iloc[:, i].to_numpy()
            y_true = df_ws_subset.iloc[:, i].to_numpy()
            imputer = SimpleImputer(strategy="mean")
            y_true_imputed = imputer.fit_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_imputed = imputer.fit_transform(y_pred.reshape(-1, 1)).flatten()
            rmse = np.sqrt(mean_squared_error(y_true_imputed, y_pred_imputed))
            rmse_list.append(rmse)
            # print(f"RMSE for column {i}:{rmse}")
        except Exception as e:
            err_count += 1

    # 计算 RMSE 的平均值
    rmse_series = pd.Series(rmse_list)
    pass

if __name__ == '__main__':
    main()