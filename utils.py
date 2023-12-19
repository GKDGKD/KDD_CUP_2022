import pandas as pd
import os, time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_and_val(data_train, data_val, model, optimizer,  criterion, config, scheduler=None, logger=None):
    loader_train = DataLoader(
        dataset=data_train, batch_size=config['batch_size'], shuffle=config['shuffle_train_val']
    )
    loader_val = DataLoader(
        dataset=data_val, batch_size=config['batch_size'], shuffle=config['shuffle_test']
    )

    model.to(config['device'])
    best_validation_loss = float('inf')
    patience_counter     = 0
    patience             = config['patience']
    train_loss_history   = []
    val_loss_history     = []
    
    for epoch in range(config['max_epoch']):
        train_loss = []
        epoch_start_time = time.time()
        model.train()
        for i, (x, y) in enumerate(loader_train):
            optimizer.zero_grad()
            x.to(config['device'])
            y.to(config['device'])
            out = model(x)
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
                x.to(config['device'])
                y.to(config['device'])
                out = model(x)
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
            logger.info('Epoch:{:d}, train_loss: {:.4f}, validation_loss: {:.4f}, cost time: {:.2f}s'.format(epoch + 1, 
                                                                                                             train_loss_epoch, 
                                                                                                             val_loss_epoch,
                                                                                                             cost_time))
        else:
            print('Epoch:{:d}, train_loss: {:.4f}, validation_loss: {:.4f}, cost time: {:.2f}s'.format(epoch + 1, 
                                                                                                             train_loss_epoch, 
                                                                                                             val_loss_epoch,
                                                                                                             cost_time))
    return train_loss_history, val_loss_history