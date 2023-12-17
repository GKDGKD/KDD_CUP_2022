import pandas as pd
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
Plan:
1. 先整个给LSTM等时序模型训练用的数据集class；
2. 先跑几个baseline
3. GNN，启动！

"""

class Scaler(object):
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        mean = torch.tensor(self.mean).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.tensor(self.std).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.tensor(self.mean) if torch.is_tensor(data) else self.mean
        std = torch.tensor(self.std) if torch.is_tensor(data) else self.std
        return (data * std) + mean

class WindTurbineDataset(Dataset):
    def __init__(self, data_path, 
                 filename='my.csv', 
                 flag='train', 
                 size=None, 
                 turbine_id=0, 
                 task='MS', 
                 target='Target',
                 scale=True, 
                 start_col=2, 
                 day_len=24 * 6, 
                 train_days=15, 
                 val_days=3, 
                 test_days=6, 
                 total_days=30):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.task = task
        self.target = target
        self.scale = scale
        self.start_col = start_col
        self.data_path = data_path
        self.filename = filename
        self.tid = turbine_id
        self.total_size = self.unit_size * total_days
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.test_size = test_days * self.unit_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = Scaler()
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        border1s = [self.tid * self.total_size,
                    self.tid * self.total_size + self.train_size - self.input_len,
                    self.tid * self.total_size + self.train_size + self.val_size - self.input_len
                    ]
        border2s = [self.tid * self.total_size + self.train_size,
                    self.tid * self.total_size + self.train_size + self.val_size,
                    self.tid * self.total_size + self.train_size + self.val_size + self.test_size
                    ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw
        if self.task == 'M':
            cols_data = df_raw.columns[self.start_col:]
            df_data = df_raw[cols_data]
        elif self.task == 'MS':
            cols_data = df_raw.columns[self.start_col:]
            df_data = df_raw[cols_data]
        elif self.task == 'S':
            df_data = df_raw[[self.tid, self.target]]

        pd.set_option('mode.chained_assignment', None)
        df_data.replace(to_replace=np.nan, value=0, inplace=True)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.raw_data = df_data[border1 + self.input_len:border2]

    def __getitem__(self, index):
        if self.set_type >= 3:
            index = index * self.output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = torch.from_numpy(self.data_x[s_begin:s_end])
        seq_y = torch.from_numpy(self.data_y[r_begin:r_end])
        return seq_x, seq_y

    def __len__(self):
        if self.set_type < 3:
            return len(self.data_x) - self.input_len - self.output_len + 1
        return int((len(self.data_x) - self.input_len) / self.output_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# 使用示例
data_path = 'your_data_path'
paddle_dataset = WindTurbineDataset(data_path=data_path, flag='train')
pytorch_dataset = paddle_dataset
pytorch_dataloader = torch.utils.data.DataLoader(pytorch_dataset, batch_size=64, shuffle=True)
