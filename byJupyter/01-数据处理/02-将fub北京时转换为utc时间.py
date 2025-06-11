from utils.common import get_realdata_df


def main():
    read_file_path: str = r'/Volumes/DATA/FUB/MF01001/2024_local.csv'
    out_put_file_path: str = r'/Volumes/DATA/FUB/MF01001/2024_local_df_utc.csv'
    # step1: 读取原始整年数据
    # step2: 按照时间步长生成对应的dataframe
    df_utc = get_realdata_df(read_file_path)
    # step3: 将生成的 df 存储为 .csv 文件
    df_utc.to_csv(out_put_file_path)

    pass


if __name__ == '__main__':
    main()
