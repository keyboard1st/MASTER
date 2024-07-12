import pandas as pd
from utils import get_foldpq, read_component
import os


def data_concat():
    folder_path = get_foldpq()

    parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pq')]

    dataframes = []
    batch_size = 5  # 根据实际内存情况调整批次大小

    cp = read_component()
    for i in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[i:i + batch_size]
        batch_df = pd.concat((pd.read_parquet(file) for file in batch_files), axis=0)
        select_result = pd.DataFrame(columns=batch_df.columns,
                          index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=batch_df.index.names))
        for date in cp.index:
            # 检查日期是否存在于df1的索引中
            if date not in batch_df.index.levels[0]:
                continue
            hs300_stocks = cp.loc[date].dropna().index.tolist()
            date_data = batch_df.loc[(date, hs300_stocks), :]
            select_result = pd.concat([select_result, date_data])

        dataframes.append(select_result)

    combined_df = pd.concat(dataframes, axis=0)

    combined_df.to_parquet('combined_data.pq', engine='pyarrow')

    print("********** data concat done **********")


if __name__ == '__main__':
    data_concat()
    df = pd.read_parquet('combined_data.pq')
    print(df)
    print(df.isna().sum())

