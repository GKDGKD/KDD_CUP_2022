import time, os
import torch
import torch.nn as nn
from log.logutli import Logger
from utils import train_and_val
from models import RNN
from data_prepare import WindTurbineDataset

def run():
    data_path  = './data/train/'
    filename   = 'wtbdata_245days.csv'
    flag       = 'train'
    input_len  = 288
    output_len = 288
    size       = [input_len, output_len]
    task       = 'MS'
    target     = 'Patv'
    start_col  = 3
    turbine_id = 0
    day_len    = 144
    train_days = 240
    val_days   = 3
    test_days  = 2
    total_days = 245

    data_train = WindTurbineDataset(
        data_path  = data_path,
        filename   = filename,
        flag       = flag,
        size       = size,
        task       = task,
        target     = target,
        start_col  = start_col,
        turbine_id = turbine_id,
        day_len    = day_len,
        train_days = train_days,
        val_days   = val_days,
        test_days  = test_days,
        total_days = total_days
    )

    data_val = WindTurbineDataset(
        data_path  = data_path,
        filename   = filename,
        flag       = 'val',
        size       = size,
        task       = task,
        target     = target,
        start_col  = start_col,
        turbine_id = turbine_id,
        day_len    = day_len,
        train_days = train_days,
        val_days   = val_days,
        test_days  = test_days,
        total_days = total_days
        )

    print(f'len(data_train) = {len(data_train)}, len(data_val) = {len(data_val)}')
    config = {
        'input_size'       : 10,
        'hidden_size'      : 32,
        'output_size'      : output_len,
        'num_layers'       : 1,
        'device'           : 'cuda',
        'batch_size'       : 32,
        'lr_rate'          : 0.01,
        'max_epoch'        : 100,
        'patience'         : 5,
        'shuffle_train_val': True,
        'shuffle_test'     : False,
        'lr_step_size'     : 30,
        'lr_gamma'         : 0.9
    }

    criterion = nn.MSELoss(reduction='mean')

    model_rnn = RNN(input_size=config['input_size'], hidden_size=32, output_size=output_len)

    train_and_val(data_train, data_val, model_rnn, criterion, config)

run()

def main():

    # Logger
    start_time   = time.time()
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time))
    save_dir     = os.path.join("result", current_time)
    log_id       = 'main'
    log_name     = f'Run_{current_time}.log'
    log_level    = 'info'
    logger       = Logger(log_id, save_dir, log_name, log_level)
    logger.logger.info(f"LOCAL TIME: {current_time}")
    
    # 结果保存路径
    save_dir_single = os.path.join(save_dir, 'Single')
    if not os.path.exists(save_dir_single):
        os.makedirs(save_dir_single)