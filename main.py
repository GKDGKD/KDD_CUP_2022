import time, os
import json
import torch
import torch.nn as nn
from log.logutli import Logger
from models import RNN
from data_prepare import WindTurbineDataset
from train import traverse_wind_farm

def main():

    # Logger
    start_time   = time.time()
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time))
    save_dir     = os.path.join("result", current_time)
    log_id       = 'main'
    log_name     = f'Run_{current_time}.log'
    log_level    = 'info'
    Logger_      = Logger(log_id, save_dir, log_name, log_level)
    logger       = Logger_.logger
    logger.info(f"LOCAL TIME: {current_time}")
    
    # 结果保存路径
    save_dir_model = os.path.join(save_dir, 'model')
    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)

    # 读取参数
    with open('./config.json', 'r') as f:
        config = json.load(f)

    model_rnn = RNN(input_size=config['input_size'], 
                    hidden_size=config['hidden_size'], 
                    output_size=config['output_len'],
                    num_layers=config['num_layers'])
    criterion = nn.MSELoss(reduction='mean')
    traverse_wind_farm(model_rnn, criterion, config, save_dir_model, logger)

if __name__ == "__main__":
    main()