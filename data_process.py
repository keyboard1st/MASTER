from qlib.data.dataset.processor import RobustZScoreNorm, Fillna, DropnaLabel, CSZScoreNorm
import pandas as pd


def read_pq():
    return pd.read_parquet("D:\CX_code\Graph series\MASTER\\utils\combined_data.pq")


def set_multi_index(df):
    level0 = ['feature'] * 44 + ['label']  # 第一层索引：前44列为'feature'，最后一列为'label'
    level1 = list(df.keys())
    multi_index = pd.MultiIndex.from_tuples(list(zip(level0, level1)))
    df.columns = multi_index
    df.index.names = ['datetime', 'instrument']
    return df


def sort_index(df):
    return df.sort_index()


def processor(df):
    robust_zscore_norm = RobustZScoreNorm(fit_start_time='20160101', fit_end_time='20240101', fields_group='feature',
                                          clip_outlier=True)
    fillna = Fillna(fields_group='feature')
    dropna_label = DropnaLabel()
    cszscore_norm = CSZScoreNorm(fields_group='label')

    robust_zscore_norm.fit(df)
    for process_func in [robust_zscore_norm, fillna, cszscore_norm]:
        df_processed = process_func(df)
    return df_processed


def data_processor():
    # step1:获取数据并生成复制
    df = read_pq()
    # df = read_pq(combined_datafile)
    df_test = df.copy()

    # step2:设置双重索引
    df_test = set_multi_index(df_test)

    # step3:索引排序
    # print(df_test.index)
    df_sorted = sort_index(df_test)
    # print(df_sorted.index)

    # step4:数据预处理
    df_processed = processor(df_sorted)

    # print(df_sorted.isna().sum())
    # print(df_process.isna().sum())
    # print(df_sorted.head())
    # print(df_process.head())

    # feature_mean = df_process['label'].mean()
    # feature_std = df_process['label'].std()

    # print(f"Mean of 'feature_column' after preprocessing: {feature_mean}")
    # print(f"Standard deviation of 'feature_column' after preprocessing: {feature_std}")

    print("********** data process done **********")
    return df_processed

if __name__ == '__main__':
    df = data_processor()
    print(df)
