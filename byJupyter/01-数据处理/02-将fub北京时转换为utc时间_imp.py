from utils.common import get_realdata_df


def main():
    read_file_path: str = r'Z:/FUB/MF01001/2024_local.csv'
    out_put_file_path: str = r'Z:/FUB/MF01001/2024_local_df_utc_183.csv'
    # step1: 读取原始整年数据
    # step2: 按照时间步长生成对应的dataframe
    # TODO:[-] 25-06-08 GRAPES 风场数据时间步长为 3h , 预报时次为 61 。每组总计需要取 3*61=183 个实况
    df_utc = get_realdata_df(read_file_path, 183)
    # step3: 将生成的 df 存储为 .csv 文件
    df_utc.to_csv(out_put_file_path)

    pass


if __name__ == '__main__':
    main()
