import pickle
import os
import pandas as pd
from qlib.data.dataset import TSDataSampler
from data_process import data_processor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# 划分数据集
def split_data(df, train_start, train_end, val_start, val_end, test_start, test_end):

    train_range = pd.date_range(train_start, train_end)
    val_range = pd.date_range(val_start, val_end)
    test_range = pd.date_range(test_start, test_end)

    idx = pd.IndexSlice
    train_set = df.loc[idx[train_range, :], :]
    val_set = df.loc[idx[val_range, :], :]
    test_set = df.loc[idx[test_range, :], :]

    return train_set, val_set, test_set

def build_dataset(df, start, end):
    """
    df: pd.DataFrame
    start and end: str
    """
    dataset = TSDataSampler(df, start, end, step_len=8, fillna_type="ffill+bfill")

    return dataset

def get_all_dataset(df, train_start, train_end, val_start, val_end, test_start, test_end):

    train_set, val_set, test_set = split_data(df, train_start, train_end, val_start, val_end, test_start, test_end)

    train_dataset = build_dataset(train_set, train_start, train_end)
    val_dataset = build_dataset(val_set, val_start, val_end)
    test_dataset = build_dataset(test_set, test_start, test_end)

    return train_dataset, val_dataset, test_dataset

def data_provider():
    df = data_processor()
    dl_train, dl_valid, dl_test = get_all_dataset(df, '20160101', '20230831',
                                                  '20230901', '20231231',
                                                  '20240101', '20240601')
    return dl_train, dl_valid, dl_test

if __name__ == '__main__':
    dl_train, dl_valid, dl_test = data_provider()
    # 定义保存路径
    save_path = "D:\CX_code\Graph series\MASTER\dataset\Alpha57"

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        # 保存dl_train
    with open(os.path.join(save_path, 'Alpha57_dl_train.pkl'), 'wb') as f:
        pickle.dump(dl_train, f)

        # 保存dl_valid
    with open(os.path.join(save_path, 'Alpha57_dl_valid.pkl'), 'wb') as f:
        pickle.dump(dl_valid, f)

        # 保存dl_test
    with open(os.path.join(save_path, 'Alpha57_dl_test.pkl'), 'wb') as f:
        pickle.dump(dl_test, f)

    print("********** data dump done **********")





