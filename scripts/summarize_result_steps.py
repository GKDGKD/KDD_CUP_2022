import os
import pandas as pd
import re
from tqdm import tqdm

# 设置结果文件夹路径
result_folder = './result'
save_dir = './result'

# 删除旧文件
save_path = os.path.join(save_dir, 'summary_steps.xlsx')
if os.path.exists(save_path):
    os.remove(save_path)


# 遍历结果文件夹
date_history = []
result_all = pd.DataFrame()
for date_folder in tqdm(os.listdir(result_folder)):
    date_folder_path = os.path.join(result_folder, date_folder)
    date_part        = date_folder.split('_')
    date             = "_".join(date_part[:-1])
    model_name       = date_part[-1].strip()
    date_history.append(date)
    
    result_csv_path = os.path.join(date_folder_path, 'Regression_metrics_all_time_steps.csv')
    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
        # result_df['Step'] = result_df.iloc[:, 0].str.extract('(\d+)').fillna(0).astype(int)
        result_df.insert(0, 'Step', result_df.iloc[:, 0].str.extract('(\d+)').fillna(0).astype(int))
        result_df = result_df[result_df['Step'] != 0]
        result_df.drop(columns=['Unnamed: 0'], inplace=True)
        result_df.insert(0, '模型', [model_name] * len(result_df))
        result_df.insert(0, 'Date', [date] * len(result_df))
        # # 删除total行
        # result_df = result_df.drop(result_df.index[result_df['Unnamed: 0'] == 'total'])
        
        result_all = pd.concat([result_all, result_df], ignore_index=True)

result_all.to_excel(os.path.join(save_dir, 'summary_steps.xlsx'), index=False)

print('Done!')
