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
from utils import get_gnn_data, generate_dataset

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
    else:
        plt.show()
    plt.close()

def is_empty_folder(path):
    return len(os.listdir(path)) == 0

def evaluate(config, result_dir, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}, Begin to evaluate...')
    result_all = []
    for i in range(config['capacity']):
        model_dir = os.path.join(result_dir, 'model', f'Turbine_{i}')
        if not os.path.exists(model_dir) or is_empty_folder(model_dir):
            logger.error(f'No model in {model_dir}, please train first!')
            break
        logger.info(f'Evaluate Turbine {i}...')
        preds, gts = forecast_one(i, config, model_dir, device)  # [N, output_timestep], (3313, 288)
        breakpoint()
        # TODO: save predictions and ground truths for each turbine
        plot_predictions(preds[0], gts[0], model_dir)
        result = regression_metric(preds / 1000, gts / 1000)
        result_all.append(result)
        logger.info(', '.join([f'{k}: {v}' for k, v in result.items()]))
        result_df = pd.DataFrame(result, index=[f'Turbine_{i}'])
        result_df.to_csv(os.path.join(model_dir, f'Turbine_{i}_regression_metrics.csv'), index=False)

    result_all_df = pd.DataFrame(result_all, 
                                 columns=result.keys(),
                                 index=[f'Turbine_{i}' for i in range(config['capacity'])])
    overall_metrics = {col: result_all_df[col].sum() for col in result_all_df.columns}
    overall_df      = pd.DataFrame(overall_metrics, index=['Total'])
    result_all_df   = pd.concat([result_all_df, overall_df], axis=0)
    result_all_df.to_csv(os.path.join(result_dir, 'Regression_metrics_all_turbines.csv'), index=True)
    logger.info(', '.join([f'{k}: {v}' for k, v in overall_metrics.items()]))
    logger.info('Evaluate finished!')

def evaluate_stgcn(config, model_dir, logger):
    # 评估STGCN模型

    _, _, test_original_data, A_wave, means, stds = get_gnn_data(config, logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    model = torch.load(os.path.join(model_dir, 'model', 'STGCN', 'model_STGCN.pt'), map_location=device)
    model.to(device)
    A_wave = A_wave.to(device)
    model.eval()

    test_indices = [(i, i + (config['input_len'] + config['output_len'])) 
           for i in range(test_original_data.shape[2] - \
                          (config['input_len'] + config['output_len']) + 1)]
    
    logger.info(f'Begin to predict on {len(test_indices)} samples...')
    preds, gts = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_indices), config['batch_size'])):
            x, y = generate_dataset(test_original_data, 
                                    test_indices[i:i + config['batch_size']],
                                    config['input_len'], 
                                    config['output_len'])
            x   = x.to(device)
            out = model(A_wave, x)
            preds.append(out.cpu().numpy())  # [N, num_nodes, output_timestep], (3025, 134, 288)
            gts.append(y.cpu().numpy()) # [N, num_nodes, output_timestep]

    preds = np.concatenate(preds, axis=0)   # [N, num_nodes, output_timestep]
    gts   = np.concatenate(gts, axis=0)     # [N, num_nodes, output_timestep]

    # 逆标准化, 默认最后一列为目标变量y
    preds = preds * stds[-1] + means[-1]
    gts   = gts * stds[-1] + means[-1]
    plot_predictions(preds[0, 0, :], gts[0, 0, :], model_dir)

    logger.info(f'Caculating regression metrics on {len(test_indices)} samples...')
    result_all = []
    for i in tqdm(range(preds.shape[1])):
        result_one = regression_metric(preds[:, i, :] / 1000, gts[:, i, :] / 1000)
        result_all.append(result_one)
    result_all_df = pd.DataFrame(result_all, 
                                 columns=result_one.keys(),
                                 index=[f'Turbine_{i}' for i in range(config['capacity'])])
    overall_metrics = {col: result_all_df[col].sum() for col in result_all_df.columns}
    overall_df      = pd.DataFrame(overall_metrics, index=['Total'])
    result_all_df   = pd.concat([result_all_df, overall_df], axis=0)
    result_all_df.to_csv(os.path.join(model_dir, 'Regression_metrics_all_turbines.csv'), index=True)
    logger.info(', '.join([f'{k}: {v}' for k, v in overall_metrics.items()]))

    # Save predictions and ground truths， 太大了，一个csv文件1GB，先不保存
    logger.info(f'Saving predictions and ground truths in {model_dir}...')
    preds   = preds.reshape(preds.shape[1] * preds.shape[0], preds.shape[2]) # [num_nodes * N, output_timestep]
    gts     = gts.reshape(gts.shape[1] * gts.shape[0], gts.shape[2])    # [num_nodes * N, output_timestep]
    pred_df = pd.DataFrame(preds, columns=[f'pred_{i + 1}' for i in range(preds.shape[1])])
    gt_df   = pd.DataFrame(gts, columns=[f'truth_{i + 1}' for i in range(gts.shape[1])])
    pred_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)
    gt_df.to_csv(os.path.join(model_dir, 'ground_truths.csv'), index=False)
    logger.info('Evaluate finished!')

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)

    model_name = config['model_name']

    # Logger
    start_time   = time.time()
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time))
    save_dir     = os.path.join("result", current_time + f'_evaluate_{model_name}')
    log_id       = 'evaluate'
    log_name     = f'Run_{current_time}.log'
    log_level    = 'info'
    Logger_      = Logger(log_id, save_dir, log_name, log_level)
    logger       = Logger_.logger
    logger.info(f"LOCAL TIME: {current_time}")

    result_dir = './result/2023_12_26_17_11_41_STGCN'
    # result_dir = './result/2023_12_22_23_50_58_GRU'
    logger.info(f'Result directory: {result_dir}')
    # evaluate(config, result_dir, logger)
    evaluate_stgcn(config, result_dir, logger)