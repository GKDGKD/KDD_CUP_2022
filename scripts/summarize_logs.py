import os
import pandas as pd
import re
from tqdm import tqdm

# 设置结果文件夹路径
result_folder = './result'
save_dir = './result'

words_to_del = ['filename', 'target', 'start_col', 'turbine_id', 'capacity', 'day_len', \
                'task', 'input_size', "(10,), stds.shape", 'train_original_data','data_path', \
        'val_original_data', 'test_original_data', "cuda, model", 'Use model', 'Loss function', \
        'flag', 'model_name', 'location_path', 'shuffle_train_val', 'shuffle_test', 'Raw data.shape', \
            'total_days', 'device']


# filename,target,start_col,turbine_id,capacity,day_len,train_days,val_days,
# test_days,total_days,hidden_size,output_len,input_len,num_layers,device,batch_size,
# lr_rate,max_epoch,patience,shuffle_train_val,shuffle_test,lr_step_size,lr_gamma,location_path,
# thresh_distance,loss_fn,delta,train_type,Raw data.shape,

words_to_use = ['Date', '模型', 'Score', 'thresh_distance', 'train_type', \
                'train_days', 'val_days', 'test_days', 'hidden_size', 'output_len',\
                    'input_len', 'num_layers', 'dropout', 'kernel_size', 'device', 'loss_fn',
                'batch_size', 'lr_rate', 'max_epoch', 'patience', 'delta', 'lr_step_size', 'lr_gamma']

# 删除旧文件
save_path = os.path.join(save_dir, 'summary.xlsx')
if os.path.exists(save_path):
    os.remove(save_path)


# 遍历结果文件夹
date_history = []
for date_folder in tqdm(os.listdir(result_folder)):
    date_folder_path = os.path.join(result_folder, date_folder)
    date_part        = date_folder.split('_')
    date             = "_".join(date_part[:-1])
    model_name       = date_part[-1].strip()
    date_history.append(date)
    
    # 检查是否是文件夹
    # if os.path.isdir(date_folder_path):
    # 读取 log 文件
    log_file_path = os.path.join(date_folder_path, f'Run_{date}.log')
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            log_text = log_file.read()
        
        # 通过正则表达式找到 Parameters 行和 Training 行之间的内容, 找参数
        match = re.search(r'Parameters:(.*?)------ Training', log_text, re.DOTALL)

        if match:
            parameters_section = match.group(1).strip()

            # 将参数内容分割成字典
            parameters_dict = {}
            for parameter_line in parameters_section.split('\n'):
                if ':' in parameter_line:
                    tmp = parameter_line.split(':')
                    if 'main-INFO' not in tmp[-2].strip():
                        parameters_dict[tmp[-2].strip()] = tmp[-1].strip()

            # # 如果需要追加到已有的 summarize.csv，使用 mode='a'，否则使用 mode='w'
            # df.to_csv('summarize.csv', mode='a', index=False, header=not os.path.exists('summarize.csv'))
        else:
            print("Pattern not found in log text.")


        # 找时间
        pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}-main-INFO: Running time: (\d+\.\d+) hours, (\d+\.\d+) minutes, (\d+\.\d+) seconds')
        match2 = pattern.search(log_text)

        if match2:
            # group(1)：hours, group(2)：minutes, group(3)：seconds
            seconds = round(float(match2.group(3)), 4)
            # print(f'seconds: {seconds}')
        else:
            seconds = -1


        # 读取 result.csv 文件的最后一列最后一行的 score
        result_csv_path = os.path.join(date_folder_path, 'Regression_metrics_all_turbines.csv')
        if os.path.exists(result_csv_path):
            result_df = pd.read_csv(result_csv_path)
            # 获取最后一列最后一行的 score
            score = result_df['Score'].iloc[-1]
            # breakpoint()
            mae   = result_df['MAE'].iloc[-1]
            rmse  = result_df['RMSE'].iloc[-1]
        else:
            score = 'N/A'  # 如果 result.csv 文件不存在，默认为 'N/A'
            mae   = 'N/A'
            rmse  = 'N/A'
        
        # 将结果添加到 summary_df 中
        summary_dict = pd.DataFrame({
            'Date'     : date,
            '模型'     : model_name,
            'MAE'      : mae,
            'RMSE'     : rmse,
            'Score'    : score,
            'cost time': seconds,
            **parameters_dict
        }, index=date_history)
        summary_df = pd.DataFrame.from_dict(summary_dict)
        summary_df = summary_df.drop(columns=words_to_del, axis=1, errors='ignore')
        # summary_df = summary_df[words_to_use]
        summary_df.drop_duplicates(subset=['Date', '模型'],inplace=True)

        # breakpoint()

        # 如果 summarize.csv 存在，读取已有的 DataFrame
        if os.path.exists(save_path):
            existing_data = pd.read_excel(save_path)
            
            # 追加新数据
            updated_data = pd.concat([existing_data, summary_df], ignore_index=True)
        else:
            updated_data = summary_df

        # 将更新后的数据保存为 summarize.csv
        updated_data.to_excel(save_path, index=False)

    else:
        print(f"Log file not found: {log_file_path}")

print('Done!')
