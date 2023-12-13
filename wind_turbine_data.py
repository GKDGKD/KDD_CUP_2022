import pandas as pd
import torch
from torch.utils.data import Dataset

class WindPowerDataset(Dataset):
    def __init__(self, csv_path_wtbdata, csv_path_location, history_days=14, forecast_days=2):
        self.history_days = history_days
        self.forecast_days = forecast_days

        # 读取风机数据
        self.wind_data = pd.read_csv(csv_path_wtbdata)
        # 读取风机地理位置数据
        self.location_data = pd.read_csv(csv_path_location)

        # 合并数据集
        self.merged_data = pd.merge(self.wind_data, self.location_data, on='TurbID')

    def __len__(self):
        return len(self.merged_data)

    def __getitem__(self, idx):
        turb_id = self.merged_data['TurbID'].iloc[idx]

        # 提取指定风机的历史功率数据
        history_data = self.merged_data[self.merged_data['TurbID'] == turb_id].tail(self.history_days)

        # 提取指定风机的地理位置数据
        location_data = self.location_data[self.location_data['TurbID'] == turb_id]

        # 提取目标变量（接下来2天的发电功率）
        forecast_data = self.merged_data[self.merged_data['TurbID'] == turb_id].head(self.forecast_days)

        # 提取历史时间序列数据
        time_series_data = self.merged_data[self.merged_data['TurbID'] == turb_id]['Patv'].values

        # 将历史时间序列数据划分为输入序列和目标序列
        input_sequence = time_series_data[:-self.forecast_days]
        target_sequence = time_series_data[-self.forecast_days:]

        # 在这里，你需要根据你的数据格式提取相应的特征和目标变量

        # 将数据转换为 PyTorch 张量
        history_data_tensor = torch.tensor(history_data.values, dtype=torch.float32)
        location_data_tensor = torch.tensor(location_data.values, dtype=torch.float32)
        input_sequence_tensor = torch.tensor(input_sequence, dtype=torch.float32)
        target_sequence_tensor = torch.tensor(target_sequence, dtype=torch.float32)

        return {
            'history_data': history_data_tensor,
            'location_data': location_data_tensor,
            'input_sequence': input_sequence_tensor,
            'target_sequence': target_sequence_tensor
        }

# 示例用法
csv_path_wtbdata = 'path/to/wtbdata_245days.csv'
csv_path_location = 'path/to/sdwpf_baidukddcup2022_turb_location.CSV'

wind_power_dataset = WindPowerDataset(csv_path_wtbdata, csv_path_location)
data_point = wind_power_dataset[0]  # 获取第一个数据点
