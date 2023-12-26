import pandas as pd
import os, time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_prepare import WindTurbineDataset
from models import RNN, LSTM, GRU, STGCN
from utils import get_gnn_data, generate_dataset

def get_train_and_val_data(config, turbine_id=0):
    data_train = WindTurbineDataset(
        data_path  = config['data_path'],
        filename   = config['filename'],
        flag       = 'train',
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

    data_val = WindTurbineDataset(
        data_path  = config['data_path'],
        filename   = config['filename'],
        flag       = 'val',
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
    
    loader_train = DataLoader(dataset=data_train, batch_size=config['batch_size'], shuffle=config['shuffle_train_val'])
    loader_val   = DataLoader(dataset=data_val, batch_size=config['batch_size'], shuffle=config['shuffle_test'])

    return data_train, data_val, loader_train, loader_val


def train_and_val(turbine_id, model, criterion, config, model_save_dir, logger=None):
    data_train, data_val, loader_train, loader_val = get_train_and_val_data(config, turbine_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if logger:
        logger.info(f'Device: {device}, len(data_train): {len(data_train)}, len(data_val): {len(data_val)}')
    else:
        print(f'Device: {device}, len(data_train): {len(data_train)}, len(data_val): {len(data_val)}')
    best_validation_loss = float('inf')
    patience_counter     = 0
    patience             = config['patience']
    train_loss_history   = []
    val_loss_history     = []
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_rate']) # 这两个不能放外面
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['lr_step_size'], 
                                                gamma=config['lr_gamma'])
    
    for epoch in range(config['max_epoch']):
        train_loss = []
        epoch_start_time = time.time()
        model.train()
        for x, y in loader_train:
            x, y = x.to(device), y.to(device)
            # print(f'x.device: {x.device}, y.device: {y.device}, model.device: {model.device}')
            optimizer.zero_grad()
            out  = model(x).to(device)
            # print(f'out.device: {out.device}, y.device:{y.device}')
            # breakpoint()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss_epoch = np.mean(train_loss)
        train_loss_history.append(train_loss_epoch)
        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = []
            for x, y in loader_val:
                x    = x.to(device)
                y    = y.to(device)
                out  = model(x).to(device)
                loss = criterion(out, y)
                val_loss.append(loss.item())
        val_loss_epoch = np.mean(val_loss)
        val_loss_history.append(val_loss_epoch)
        epoch_end_time = time.time()
        cost_time = epoch_end_time - epoch_start_time

        # 早停
        if val_loss_epoch < best_validation_loss:
            best_validation_loss = val_loss_epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if logger:
                logger.info(f'Early stopping after {patience} epochs without improvement.')
            else:
                print(f'Early stopping after {patience} epochs without improvement.')
            break
        
        if logger:
            logger.info(f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
                f'Train Loss: {train_loss_epoch:.4f}, '
                f'Validation Loss: {val_loss_epoch:.4f}, '
                f'Learning Rate: {scheduler.get_last_lr()[0]:.2e},'
                f'Cost time: {cost_time:.2f}s')
        else:
            print(f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
                f'Train Loss: {train_loss_epoch:.4f}, '
                f'Validation Loss: {val_loss_epoch:.4f}, '
                f'Learning Rate: {scheduler.get_last_lr()[0]:.2e},'
                f'Cost time: {cost_time:.2f}s')
            
    if model_save_dir:
        torch.save(model, os.path.join(model_save_dir, f'model_{turbine_id}.pt'))
        plot_loss(train_loss_history, val_loss_history, model_save_dir, turbine_id)
            
    return train_loss_history, val_loss_history

def train_stgcn(model, criterion, config, model_save_dir, logger=None):
    # 训练STGCN模型

    train_original_data, val_original_data, _, \
        A_wave, means, stds = get_gnn_data(config, logger)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
        
    best_validation_loss = float('inf')
    patience_counter     = 0
    patience             = config['patience']
    train_loss_history   = []
    val_loss_history     = []
    train_indices = [(i, i + (config['input_len'] + config['output_len'])) 
           for i in range(train_original_data.shape[2] - \
                          (config['input_len'] + config['output_len']) + 1)]
    val_indices = [(i, i + (config['input_len'] + config['output_len'])) 
           for i in range(val_original_data.shape[2] - \
                          (config['input_len'] + config['output_len']) + 1)]
    
    model.to(device)
    A_wave.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_rate']) # 这两个不能放外面
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['lr_step_size'], 
                                                gamma=config['lr_gamma'])
    
    for epoch in range(config['max_epoch']):
        train_loss = []
        epoch_start_time = time.time()
        model.train()
        for i in range(0, len(train_indices), config['batch_size']):
            x, y = generate_dataset(train_original_data, 
                                    train_indices[i:i + config['batch_size']],
                                    config['input_len'], 
                                    config['output_len'])
            x, y = x.to(device), y.to(device)
            # print(f'x.device: {x.device}, y.device: {y.device}, model.device: {model.device}')
            optimizer.zero_grad()
            out  = model(A_wave, x).to(device)
            # print(f'out.device: {out.device}, y.device:{y.device}')
            # breakpoint()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            logger.info(f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
                    f'Batch: {i}/{int(len(train_indices))}, '
                    f'Train Loss: {loss.item():.4f}')

        train_loss_epoch = np.mean(train_loss)
        train_loss_history.append(train_loss_epoch)
        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = []
            for i in range(0, len(val_indices), config['batch_size']):
                x, y = generate_dataset(val_original_data, 
                                        val_indices[i:i + config['batch_size']],
                                        config['input_len'], 
                                        config['output_len'])
                x    = x.to(device)
                y    = y.to(device)
                out  = model(A_wave, x).to(device)
                loss = criterion(out, y)
                val_loss.append(loss.item())
        val_loss_epoch = np.mean(val_loss)
        val_loss_history.append(val_loss_epoch)
        epoch_end_time = time.time()
        cost_time = epoch_end_time - epoch_start_time

        # 早停
        if val_loss_epoch < best_validation_loss:
            best_validation_loss = val_loss_epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if logger:
                logger.info(f'Early stopping after {patience} epochs without improvement.')
            else:
                print(f'Early stopping after {patience} epochs without improvement.')
            break
        
        if logger:
            logger.info(f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
                f'Train Loss: {train_loss_epoch:.4f}, '
                f'Validation Loss: {val_loss_epoch:.4f}, '
                f'Learning Rate: {scheduler.get_last_lr()[0]:.2e},'
                f'Cost time: {cost_time:.2f}s')
        else:
            print(f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
                f'Train Loss: {train_loss_epoch:.4f}, '
                f'Validation Loss: {val_loss_epoch:.4f}, '
                f'Learning Rate: {scheduler.get_last_lr()[0]:.2e},'
                f'Cost time: {cost_time:.2f}s')
            
    if model_save_dir:
        model_name = config['model_name']
        torch.save(model, os.path.join(model_save_dir, f'model_{model_name}.pt'))
        plot_loss(train_loss_history, val_loss_history, model_save_dir, config['model_name'])
            
    return train_loss_history, val_loss_history


def plot_loss(train_loss_history, val_loss_history, model_save_dir, turbine_id):
    plt.figure(figsize=(10, 6), facecolor='w')
    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, f'loss_{turbine_id}.png'))


def traverse_wind_farm(config, model_save_dir, logger=None):
    # 一个模型训多次
    model_map = {
        'rnn': RNN(input_size=config['input_size'], 
                    hidden_size = config['hidden_size'],
                    output_size = config['output_len'],
                    num_layers  = config['num_layers']),
        'lstm': LSTM(input_size=config['input_size'],
                    hidden_size = config['hidden_size'],
                    output_size = config['output_len'],
                    num_layers  = config['num_layers']),
        'gru': GRU(input_size=config['input_size'],
                    hidden_size = config['hidden_size'],
                    output_size = config['output_len'],
                    num_layers  = config['num_layers']),
        'stgcn': STGCN(num_nodes=config['capacity'],
                        num_features         = 10 if config['start_col'] == 3 else 1,
                        num_timesteps_input  = config['input_len'] ,
                        num_timesteps_output = config['output_len'])
    }

    logger.info(f'Use model: {config["model_name"]}')

    if config['model_name'].lower() == 'stgcn':
        model = model_map[config['model_name'].lower()]
        criterion = nn.MSELoss(reduction='mean')
        logger.info('-' * 30 + f' Training STGCN ' + '-' * 30)
        save_dir = os.path.join(model_save_dir, 'STGCN')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_stgcn(model, criterion, config, model_save_dir, logger)

    elif any(config['model_name'].lower() in model_map.keys()):
        for i in range(config['capacity']):
            model = model_map[config['model_name'].lower()]
            criterion = nn.MSELoss(reduction='mean')
            logger.info('-' * 30 + f' Training Turbine {i} ' + '-' * 30)
            save_dir = os.path.join(model_save_dir, f'Turbine_{i}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            train_and_val(i, model, criterion, config, save_dir, logger)

    else:
        logger.error(f'The model {config["model_name"]} is not implemented.')
    
