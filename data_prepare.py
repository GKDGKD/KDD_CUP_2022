import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


class Scaler(object):
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data, to_tensor=True):
        if to_tensor:
            return torch.tensor((data - self.mean) / self.std, dtype=torch.float32)
        else:
            return (data - self.mean) / self.std

    def inverse_transform(self, data, to_tensor=False):
        if to_tensor:
            return torch.tensor(data * self.std + self.mean, dtype=torch.float32)
        else:
            return data * self.std + self.mean

class WindTurbineDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """
    def __init__(self, data_path,
                 filename='my.csv',
                 flag='train',
                 size=None,
                 turbine_id=0,
                 task='MS',
                 target='Patv',
                 scale=True,
                 start_col=2,       # the start column index of the data one aims to utilize
                 day_len=24 * 6,
                 train_days=15,     # 15 days
                 val_days=3,        # 3 days
                 test_days=6,       # 6 days
                 total_days=30      # 30 days
                 ):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type  = type_map[flag]
        self.task      = task
        self.target    = target
        self.scale     = scale
        self.start_col = start_col
        self.data_path = data_path
        self.filename  = filename
        self.tid       = turbine_id

        # If needed, we employ the predefined total_size (e.g. one month)
        self.total_size = self.unit_size * total_days
        #
        self.train_size = train_days * self.unit_size
        self.val_size   = val_days * self.unit_size
        self.test_size  = test_days * self.unit_size
        # self.test_size = self.total_size - train_size - val_size
        #
        # Or, if total_size is unavailable:
        # self.total_size = self.train_size + self.val_size + self.test_size
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
            df_data = df_raw[[self.target]]

        # Turn off the SettingWithCopyWarning
        pd.set_option('mode.chained_assignment', None)
        # df_data.replace(to_replace=np.nan, value=0, inplace=True)
        df_data.fillna(method='bfill', inplace=True)
        df_data['Patv'] = df_data[self.target].apply(lambda x: max(0, x))

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x   = data[border1:border2]
        self.data_y   = data[border1:border2]
        self.raw_data = df_data[border1 + self.input_len:border2]

    def get_raw_data(self):
        return self.raw_data

    def __getitem__(self, index):
        #
        # Only for customized use.
        # When sliding window not used, e.g. prediction without overlapped input/output sequences
        if self.set_type >= 3:
            index = index * self.output_len
        #
        # Standard use goes here.
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[s_begin:s_end]
        self.target_col = self.raw_data.columns.get_loc(self.target)
        seq_y = self.data_y[r_begin:r_end, self.target_col]
        return seq_x, seq_y

    def __len__(self):
        # In our case, the sliding window is adopted, the number of samples is calculated as follows
        if self.set_type < 3:
            return len(self.data_x) - self.input_len - self.output_len + 1
        # Otherwise, if sliding window is not adopted
        return int((len(self.data_x) - self.input_len) / self.output_len)

    def inverse_transform(self, data):
        if data.ndim > 1:
            return self.scaler.inverse_transform(data)
        else:
            # 逆标准化y
            num_features = self.raw_data.shape[1]
            tmp = torch.ones(len(data), num_features - 1)
            tmp = torch.cat([tmp, data.reshape(-1, 1)], dim=1)
            # Note: 默认最后一列为目标变量
            y_inverse = self.scaler.inverse_transform(tmp)[:, -1] 
    
            return y_inverse