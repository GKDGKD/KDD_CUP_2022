import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import DataLoader

def get_adjency_matrix(data_path, threshold=1000):
    # 读取数据集
    df = pd.read_csv(data_path)

    # 提取坐标列
    coordinates = df[['x', 'y']].values

    # 计算欧氏距离矩阵
    distance_matrix = euclidean_distances(coordinates)

    # 将距离矩阵转换为邻接矩阵（可以根据阈值定义邻接关系）
    # threshold = 10  # 距离阈值，根据具体情况调整
    adjacency_matrix = (distance_matrix < threshold).astype(int)

    return adjacency_matrix

def get_normalized_adj(A):
    """
    度规范化邻接矩阵，有助于在图神经网络中更好地处理不同节点度之间的差异，使得信息传递更为平滑和稳定。
    Returns the degree normalized adjacency matrix.
    """

    # 将对角线元素设为1，考虑自身因素
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))

    # 计算度矩阵D
    D = np.array(np.sum(A, axis=1)).reshape((-1,))

    # 将度矩阵D中小于10e-5的元素设为10e-5，避免除零错误
    D[D <= 10e-5] = 10e-5    # Prevent infs

    # 计算度矩阵D的倒数（逆）
    diag = np.reciprocal(np.sqrt(D))

    # 度规范化矩阵A
    # \hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}} 
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    
    return A_wave

def group_data(data, col_start, group_column='TurbID'):
    # 按TurbID分组
    group_ids = sorted(data[group_column].unique())
    col_use = data.columns[col_start:]
    res = []
    for group in group_ids:
        tmp = data[data[group_column] == group][col_use].values
        res.append(tmp)

    return np.array(res)

def generate_dataset(X, input_time_steps=12, output_time_steps=3, target_col=-1, to_tensor=True):
    # 得到窗口数据
    # [(i, j)]: i: window start point, j: window end point
    indices = [(i, i + (input_time_steps + output_time_steps)) 
           for i in range(X.shape[2] - (input_time_steps + output_time_steps) + 1)] 
    
    features, target = [], []
    for i, j in indices:
        features.append(X[:, :, i:i + input_time_steps].transpose(0, 2, 1)) # 前num_timesteps_input个
        target.append(X[:, target_col, i + input_time_steps: j])  # 后num_timesteps_output个
        
    if to_tensor:
        return (torch.from_numpy(np.array(features)).to(torch.float32), 
                torch.from_numpy(np.array(target)).to(torch.float32))
    else:
        return np.array(features, target)


def get_gnn_data(config, logger):
    # 获取图形式的数据
    data = pd.read_csv(os.path.join(config['data_path'], config['filename']))
    data.fillna(method='bfill', inplace=True)
    logger.info(f'Raw data.shape: {data.shape}')
    
    # 分组
    data = group_data(data, config['start_col']) # [num_nodes, seq_len, num_features]
    data = data.transpose((0, 2, 1)) # [num_nodes, num_features, seq_len]
    
    # 划分训练集和验证集
    train_size = config['day_len'] * config['train_days']
    val_size   = config['day_len'] * config['val_days']
    test_size  = config['day_len'] * config['test_days']

    train_original_data = data[:, :, :train_size]
    val_original_data   = data[:, :, train_size:train_size + val_size]
    test_original_data  = data[:, :, train_size + val_size:]
    logger.info(f'train_original_data: {train_original_data.shape}, \n'
      f'val_original_data: {val_original_data.shape}, \n'
      f'test_original_data: {test_original_data.shape}.')

    X_train, Y_train = generate_dataset(train_original_data, 
                                        config['input_len'], 
                                        config['output_len'],
                                        config['start_col']
                                        )
    X_val, Y_val = generate_dataset(train_original_data, 
                                        config['input_len'], 
                                        config['output_len'],
                                        config['start_col']
                                        )
    X_test, Y_test = generate_dataset(train_original_data, 
                                        config['input_len'], 
                                        config['output_len'],
                                        config['start_col']
                                        )
    
    loader_train = DataLoader(list(zip(X_train, Y_train)), batch_size=config['batch_size'], shuffle=config['shuffle_train_val'])
    loader_val = DataLoader(list(zip(X_val, Y_val)), batch_size=config['batch_size'], shuffle=config['shuffle_train_val'])
    loader_test = DataLoader(list(zip(X_test, Y_test)), batch_size=config['batch_size'], shuffle=False)

    # load adjency matrix
    A      = get_adjency_matrix(config['location_path'], config['thresh_distance'])
    A_wave = torch.from_numpy(get_normalized_adj(A)).to(torch.float32)

    return loader_train, loader_val, loader_test, A_wave