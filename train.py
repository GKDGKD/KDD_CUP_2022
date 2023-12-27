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

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_model.pt'):
        self.patience   = patience
        self.delta      = delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.path       = path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model, self.path)

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
    # 训练单个风机，采用paddle的数据集格式
    data_train, data_val, loader_train, loader_val = get_train_and_val_data(config, turbine_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if logger:
        logger.info(f'Device: {device}, len(data_train): {len(data_train)}, len(data_val): {len(data_val)}')
    else:
        print(f'Device: {device}, len(data_train): {len(data_train)}, len(data_val): {len(data_val)}')

    patience             = config['patience']
    train_loss_history   = []
    val_loss_history     = []
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_rate']) # 这两个不能放外面
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['lr_step_size'], 
                                                gamma=config['lr_gamma'])
    early_stopping = EarlyStopping(config['patience'], 
                               delta=config['delta'], 
                               path=os.path.join(model_save_dir, f'model_{turbine_id}.pt'))
    
    for epoch in range(config['max_epoch']):
        train_loss = []
        epoch_start_time = time.time()
        model.train()
        for x, y in loader_train:
            x, y = x.to(device), y.to(device)
            logger.info(f'x.shape: {x.shape}, y.shape: {y.shape}')
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

        logger.info(f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
                f'Train Loss: {train_loss_epoch:.4f}, '
                f'Validation Loss: {val_loss_epoch:.4f}, '
                f'Learning Rate: {scheduler.get_last_lr()[0]:.2e}, '
                f'Cost time: {cost_time:.2f}s')
        
        # 早停
        early_stopping(val_loss_epoch, model)
        if early_stopping.early_stop:
            logger.info(f'Early stopping after {patience} epochs without improvement.')
            break
            
    plot_loss(train_loss_history, val_loss_history, model_save_dir, turbine_id)
            
    return train_loss_history, val_loss_history

def train(model_map, criterion, config, model_save_dir, logger):
    """
    普通深度学习模型的训练函数. 一个风机一个模型
    Args:
        model_map: 模型匹配字典, dict, key: 模型名字, value: 模型类
        criterion: 损失函数, nn.Module
        config: 配置文件, dict
        model_save_dir: 模型保存路径
        logger: 日志
    Return:
        train_loss_history: 训练集损失
        val_loss_history: 验证集损失
    """
    assert config['train_type'].lower() in ['each', 'one'], f'Invalid train_type: {config["train_type"]}'
    train_original_data, val_original_data, _, _, _, _ = get_gnn_data(config, logger)  
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = config['model_name']
    patience   = config['patience']
    logger.info(f'Device: {device}, model: {model_name}')
        
    train_loss_history, val_loss_history   = [], []
    # train_original_data:  [num_nodes, num_features, seq_len] (134, 10, 28800),
    train_indices = [(i, i + (config['input_len'] + config['output_len'])) 
           for i in range(train_original_data.shape[2] - \
                          (config['input_len'] + config['output_len']) + 1)]
    val_indices = [(i, i + (config['input_len'] + config['output_len'])) 
           for i in range(val_original_data.shape[2] - \
                          (config['input_len'] + config['output_len']) + 1)]
    
    if config['train_type'].lower() == 'one':
        # 一个模型遍历所有风机
        model = model_map[config['model_name'].lower()]
        save_dir = model_save_dir

    for turbine_id in range(config['capacity']):
        logger.info('-' * 20 + f' Training Turbine {turbine_id + 1} ' + '-' * 20)
        train_data, val_data = train_original_data[turbine_id], val_original_data[turbine_id]
        if config['train_type'].lower() == 'each':
            # 每个风机一个模型
            model = model_map[config['model_name'].lower()]
            save_dir = os.path.join(model_save_dir, f'Turbine_{turbine_id}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=config['lr_step_size'], 
                                                    gamma=config['lr_gamma'])
        early_stopping = EarlyStopping(config['patience'], 
                                delta=config['delta'], 
                                path=os.path.join(save_dir, f'model_{model_name}.pt'))

        for epoch in range(config['max_epoch']):
            train_loss = []
            epoch_start_time = time.time()
            model.train()
            for i in range(0, len(train_indices), config['batch_size']):
                x, y = generate_dataset(train_data, 
                                        train_indices[i:i + config['batch_size']],
                                        config['input_len'], 
                                        config['output_len'])
                # x: [batch_size, input_len, num_features], y: [batch_size, output_len]
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
                if i % 10 == 0:
                    logger.info(f'Turbine: {turbine_id+1}/{config["capacity"]}, '
                            f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
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
                    x, y = generate_dataset(val_data, 
                                            val_indices[i:i + config['batch_size']],
                                            config['input_len'], 
                                            config['output_len'])
                    x, y = x.to(device), y.to(device)
                    out  = model(x).to(device)
                    loss = criterion(out, y)
                    val_loss.append(loss.item())
            val_loss_epoch = np.mean(val_loss)
            val_loss_history.append(val_loss_epoch)
            epoch_end_time = time.time()
            cost_time = epoch_end_time - epoch_start_time # batch_size为64时，1个epoch大概6min+
            logger.info(f'Turbine: {turbine_id+1}/{config["capacity"]}, '
                    f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
                    f'Train Loss: {train_loss_epoch:.4f}, '
                    f'Validation Loss: {val_loss_epoch:.4f}, '
                    f'Learning Rate: {scheduler.get_last_lr()[0]:.2e},'
                    f'Cost time: {cost_time:.2f}s')

            # 早停
            early_stopping(val_loss_epoch, model)
            if early_stopping.early_stop:
                logger.info(f'Early stopping after {patience} epochs without improvement.')
                break

        plot_loss(train_loss_history, val_loss_history, save_dir, config['model_name'])

    logger.info('Training is Done!')
    

def train_stgcn(model, criterion, config, model_save_dir, logger=None):
    # 训练STGCN模型
    train_original_data, val_original_data, _, \
        A_wave, means, stds = get_gnn_data(config, logger)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    train_loss_history, val_loss_history    = [], []
    train_indices = [(i, i + (config['input_len'] + config['output_len'])) 
           for i in range(train_original_data.shape[2] - \
                          (config['input_len'] + config['output_len']) + 1)]
    val_indices = [(i, i + (config['input_len'] + config['output_len'])) 
           for i in range(val_original_data.shape[2] - \
                          (config['input_len'] + config['output_len']) + 1)]
    
    model.to(device)
    A_wave = A_wave.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_rate']) # 这两个不能放外面
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['lr_step_size'], 
                                                gamma=config['lr_gamma'])
    early_stopping = EarlyStopping(config['patience'], 
                               delta=config['delta'], 
                               path=os.path.join(model_save_dir, f'model_STGCN.pt'))
    
    for epoch in range(config['max_epoch']):
        train_loss = []
        epoch_start_time = time.time()
        model.train()
        for i in range(0, len(train_indices), config['batch_size']):
            x, y = generate_dataset(train_original_data, 
                                    train_indices[i:i + config['batch_size']],
                                    config['input_len'], 
                                    config['output_len'])
            # x: [batch_size, num_nodes, input_len, num_features], [64, 134, 288, 10]
            # y: [batch_size, num_nodes, output_len], [64, 134, 288]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(A_wave, x).to(device)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if i % 10 == 0:
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
        cost_time = epoch_end_time - epoch_start_time # batch_size为64时，1个epoch大概6min+
        logger.info(f'Epoch: {epoch + 1}/{config["max_epoch"]}, '
                f'Train Loss: {train_loss_epoch:.4f}, '
                f'Validation Loss: {val_loss_epoch:.4f}, '
                f'Learning Rate: {scheduler.get_last_lr()[0]:.2e},'
                f'Cost time: {cost_time:.2f}s')

        # 早停
        early_stopping(val_loss_epoch, model)
        if early_stopping.early_stop:
            logger.info(f'Early stopping after {config["patience"]} epochs without improvement.')
            break

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
    """
    两种模型训练方式：
    1. 一个风机一个模型；
    2. 一个模型遍历所有风机。
    这里先采用第一种方式。
    """
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
    if config['loss_fn'].lower() == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif config['loss_fn'].lower() == 'huber':
        criterion = nn.HuberLoss(reduction='mean')
    elif config['loss_fn'].lower() == 'mae':
        criterion = nn.L1Loss(reduction='mean')
    else:
        raise ValueError(f'Unsupported loss function: {config["loss_fn"]}')
    logger.info(f'Loss function: {config["loss_fn"]}')

    if config['model_name'].lower() == 'stgcn':
        model = model_map[config['model_name'].lower()]
        logger.info('-' * 30 + f' Training STGCN ' + '-' * 30)
        save_dir = os.path.join(model_save_dir, 'STGCN')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_stgcn(model, criterion, config, save_dir, logger)

    elif config['model_name'].lower() in model_map.keys():
        # for i in range(config['capacity']):
        #     model = model_map[config['model_name'].lower()]
        #     logger.info('-' * 30 + f' Training Turbine {i} ' + '-' * 30)
        #     save_dir = os.path.join(model_save_dir, f'Turbine_{i}')
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     train_and_val(i, model, criterion, config, save_dir, logger)
        train(model_map, criterion, config, model_save_dir, logger)

    else:
        logger.error(f'The model {config["model_name"]} is not implemented.')
    
