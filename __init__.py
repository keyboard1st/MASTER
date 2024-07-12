import pickle

import pandas as pd


def read_component():
    return pd.read_parquet("D:\CX_code\Graph series\MASTER\data\\000300.XSHG_component.pq")


def get_foldpq():
    folder_path = 'D:\CX_code\Graph series\MASTER\data\\alpha_57'
    return folder_path


def get_data(universe):

    with open(f'dataset/{universe}/{universe}_dl_train.pkl', 'rb') as f:
        dl_train = pickle.load(f)
    with open(f'dataset/{universe}/{universe}_dl_valid.pkl', 'rb') as f:
        dl_valid = pickle.load(f)
    with open(f'dataset/{universe}/{universe}_dl_test.pkl', 'rb') as f:
        dl_test = pickle.load(f)

    return dl_train, dl_valid, dl_test

