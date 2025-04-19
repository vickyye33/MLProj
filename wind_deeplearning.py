import csv
import pandas as pd
import numpy as np
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


def batch_readncfiles(read_path: str):
    """
        根据指定路径遍历该路径下的所有文件，并读取每个文件中的[0,23]h的时序数据(根据经纬度)
    :param read_path:
    :return:
    """
    df_nc = None
    nc_path = pathlib.Path(read_path)
    for file in nc_path.iterdir():
        # for file in 'GRAPES_2024010100_240h_UV.nc','GRAPES_2024010112_240h_UV.nc':
        if file.is_file():
            # step1: 拼接成文件全路径
            # file_full_path = nc_path / file
            file_full_path_str: str = str(file)
            # step2: 使用 xarray.open_dataset 打开 netcdf文件
            temp_df: xr.Dataset = xr.open_dataset(file_full_path_str)
            # 注意: 打开的 Dataset 有三个维度,目前只需要按照经纬度提取几个位置的24h内的时序数据
            if 'df_nc' != None:
                combined_df = pd.concat([df_nc, temp_df], ignore_index=True)
                df_nc = combined_df
            else:
                df_nc = temp_df


def main():
    read_path: str = r'Z:\风场数据\GRAPES\2024'
    batch_readncfiles(read_path)
    pass


if __name__ == '__main__':
    main()
