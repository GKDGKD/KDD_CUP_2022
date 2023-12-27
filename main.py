import time, os
import json
import torch
import torch.nn as nn
from log.logutli import Logger
from models import RNN
from data_prepare import WindTurbineDataset
from train import traverse_wind_farm
from evaluate import evaluate, evaluate_stgcn, evaluate_all

def main():

    # 读取参数
    with open('./config.json', 'r') as f:
        config = json.load(f)

    model_name = config['model_name']

    # Logger
    start_time   = time.time()
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time))
    save_dir     = os.path.join("result", current_time + f'_{model_name}')
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

    logger.info('Parameters:')
    for k, v in config.items():
        logger.info(f'{k}: {v}')

    # train
    logger.info('Starts training...')
    traverse_wind_farm(config, save_dir_model, logger)

    # evaluate
    logger.info('Starts evaluating...')
    if config['model_name'].lower() == 'stgcn':
        evaluate_stgcn(config, save_dir, logger)
    else:
        evaluate_all(config, save_dir, logger)

    cost_time = time.time() - start_time
    logger.info(f'Running time: {round(cost_time / 3600, 2)} hours, {round(cost_time / 60, 2)} minutes, '
                f'{round(cost_time, 2)} seconds.')

if __name__ == "__main__":
    main()