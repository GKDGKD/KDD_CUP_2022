import json
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_prepare import WindTurbineDataset
from metrics import regression_metric
from log.logutli import Logger

def forecast_one(turbine_id, config, model_dir, device):
    data_test = WindTurbineDataset(
        data_path  = config['data_path'],
        filename   = config['filename'],
        flag       = 'test',
        size       = [config['input_len'], config['output_len']],
        task       = config['task'],
        target     = config['target'],
        start_col  = config['start_col'],
        turbine_id = turbine_id,
        day_len    = config['day_len'],
        train_days = config['train_days'],
        val_days   = config['val_days'],
        test_days  = config['test_days'],
        total_days = config['total_days']
    )

    loader_test = DataLoader(dataset=data_test, batch_size=config['batch_size'], shuffle=False)
    
    model_dir_one = os.path.join(model_dir, f'model_{turbine_id}.pt')
    model = torch.load(model_dir_one, map_location=device)
    model.to(device)
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader_test:
            x = x.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            gts.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N, L)
    gts = np.concatenate(gts, axis=0)

    # 逆标准化
    # breakpoint()
    preds = data_test.inverse_transform(preds)
    gts = data_test.inverse_transform(gts)
    
    return preds, gts

def plot_predictions(preds, gts, savedir=None):
    assert len(preds) == len(gts)
    plt.figure(figsize=(10, 6), facecolor='w')
    plt.plot(gts, label='ground truth')
    plt.plot(preds, label='prediction')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Power')
    if savedir:
        plt.savefig(os.path.join(savedir, f'predictions.png'), dpi=200)

def is_empty_folder(path):
    return len(os.listdir(path)) == 0

def traverse_wind_farm(config, result_dir, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}, Begin to evaluate...')
    result_all = []
    for i in range(config['capacity']):
        model_dir = os.path.join(result_dir, 'model', f'Turbine_{i}')
        if not os.path.exists(model_dir) or is_empty_folder(model_dir):
            logger.error(f'No model in {model_dir}, please train first!')
            break
        logger.info(f'Evaluate Turbine {i}...')
        preds, gts = forecast_one(i, config, model_dir, device)
        result = regression_metric(preds, gts)
        result_all.append(result)
        logger.info(', '.join([f'{k}: {v}' for k, v in result.items()]))
        result_df = pd.DataFrame(result, index=[f'Turbine_{i}'])
        result_df.to_csv(os.path.join(model_dir, f'Turbine_{i}_regression_metrics.csv'), index=False)

    result_all_df = pd.DataFrame(result_all, index=[f'Turbine_{i}' for i in range(config['capacity'])])
    result_all_df.to_csv(os.path.join(result_dir, 'Regression_metrics_all_turbines.csv'), index=False)
    logger.info('Evaluate finished!')

    
if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Logger
    start_time   = time.time()
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time))
    save_dir     = os.path.join("result", current_time + '_evaluate')
    log_id       = 'evaluate'
    log_name     = f'Run_{current_time}.log'
    log_level    = 'info'
    Logger_      = Logger(log_id, save_dir, log_name, log_level)
    logger       = Logger_.logger
    logger.info(f"LOCAL TIME: {current_time}")

    result_dir = './result/2023_12_20_16_00_04'
    logger.info(f'Result directory: {result_dir}')
    traverse_wind_farm(config, result_dir, logger)