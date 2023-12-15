import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
Plan:
1. 先整个给LSTM等时序模型训练用的数据集class；
2. 先跑几个baseline
3. GNN，启动！

"""

class WindDatasetDL(Dataset):
    def __init__(self, path_train, 
                 path_location, 
                 path_test, 
                 label_y,
                 window_len,
                 scaler='minmax',
                 ):
        super().__init__()
        self.path_train    = path_train
        self.path_location = path_location
        self.path_test     = path_test
        self.label_y       = label_y
        self.window_len    = window_len
        if scaler.lower() == 'minmax':
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        elif scaler.lower() == 'standard':
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
        else:
            raise ValueError('scaler must be one of minmax or standard.')

        self.__read_data__()

    def __read_data__(self):
        # 读取数据并转换为时间序列格式
        df_train    = pd.read_csv(self.path_train)
        df_location = pd.read_csv(self.path_location)
        df_test     = pd.read_csv(self.path_test)
        df_train    = pd.merge(df_train, df_location, on='TurbID', how='left')
        df_train.replace(np.nan, value=0, inplace=True)
        X_train = df_train.drop(columns=[self.label_y])
        Y_train = df_train[self.label_y]
        
        if self.scaler:
            X_train = self.scaler_x.fit_transform(X_train)
            Y_train = self.scaler_y.fit_transform(Y_train)

            
        # 按 TurbID 分组并对每个 TurbID 进行时间序列划分
        sequences = []
        for _, group in df_train.groupby('TurbID'):
            # 按时间戳排序
            group = group.sort_values(by='Tmstamp')

            # 选择适当的历史窗口和预测窗口大小
            history_window_size = 10
            prediction_window_size = 1

            for i in range(len(group) - history_window_size - prediction_window_size + 1):
                history = group.iloc[i:i+history_window_size][['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'x', 'y']].values
                target = group.iloc[i+history_window_size:i+history_window_size+prediction_window_size]['Patv'].values
                sequences.append((torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))
