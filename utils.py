import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import eigs


def get_adjency_matrix(data_path, threshold=1000):
    # 获取邻接矩阵
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

def generate_dataset(X, 
                     indices, 
                     input_time_steps=12, 
                     output_time_steps=3, 
                     return_type = 1,
                     target_col=-1, 
                     to_tensor=True):
    """
    生成适用于STGCN的窗口格式数据集。
    Args:
        X: 原始数据，shape: [num_nodes, num_features, seq_len] or [num_features, seq_len]
        indices: window 起始点集合，[(i, j)]: i: window start point, j: window end point
        input_time_steps: input time steps, 注意两个time_step过百很容易爆内存
        output_time_steps: output time steps
        target_col: target column，默认最后一列为目标变量
        to_tensor: 是否转换为tensor
    Returns:
        features: shape: [num_windows, input_time_steps, num_nodes, num_features]
        target: shape: [num_windows, batch_size, num_nodes, output_time_steps]
    """
    
    # 得到窗口数据，注意两个time_step过百很容易爆内存
    # [(i, j)]: i: window start point, j: window end point
    # indices_all = [(i, i + (input_time_steps + output_time_steps)) 
    #        for i in range(X.shape[2] - (input_time_steps + output_time_steps) + 1)] 
    # indices = indices_all[index:index + batch_size]
    
    features, target = [], []
    if X.ndim == 3:  # [num_nodes, num_features, seq_len]
        for i, j in indices:
            features.append(X[:, :, i:i + input_time_steps].transpose(0, 2, 1)) # [batch_size, num_nodes, input_time_steps, num_features]
            target.append(X[:, target_col, i + input_time_steps: j])  # [batch_size, num_nodes, output_time_steps]
    elif X.ndim == 2:  # [num_features, seq_len]
        for i, j in indices:
            features.append(X[:, i:i + input_time_steps].transpose(1, 0)) # [batch_size, input_time_steps, num_features]
            target.append(X[target_col, i + input_time_steps: j])  # [batch_size, output_time_steps]
    elif X.ndim == 1:  # [seq_len,] 单变量时间序列，适合ARIMA
        for i, j in indices:
            features.append(X[i:i + input_time_steps]) # [N, input_time_steps]
            target.append(X[i + input_time_steps: j])  # [N, output_time_steps]
    else:
        raise ValueError('X must be 2D or 3D array')
    
    features, target = np.array(features), np.array(target)
                
    if return_type == 1:
        # STGCN input x: [batch_size, num_nodes, seq_len, num_features], [64, 134, 288, 10]
        pass
    elif return_type == 2:
        # MTGNN input x: [batch size, num_features, num_nodes, seq_len]
        features = features.transpose(0, 3, 1, 2)
        target = target.transpose(0, 2, 1)
    elif return_type == 3:
        # ASTGCN input x: [batch size, num_nodes, num_features, in_seq_len]
        # out: [batch size, num_nodes, out_seq_len]
        features = features.transpose(0, 1, 3, 2)

    if to_tensor:
        return (torch.from_numpy(features).to(torch.float32), 
                torch.from_numpy(target).to(torch.float32))
    else:
        return features, target


def get_gnn_data(config, logger):
    """
    获取训练集、验证集、测试集、邻接矩阵、特征均值、方差。
    Args:
        config: 配置文件
        logger: 日志记录器
    Returns:
        train_original_data: 训练集, shape: [num_nodes, num_features, seq_len], ndarray;
        val_original_data: 验证集, shape: [num_nodes, num_features, seq_len], ndarray;
        test_original_data: 测试集, shape: [num_nodes, num_features, seq_len], ndarray;
        A_wave: 邻接矩阵, shape: [num_nodes, num_nodes], ndarray;
        means: 特征均值, shape: [num_features, ], ndarray;
        stds: 特征方差, shape: [num_features, ], ndarray.
    """
    
    data = pd.read_csv(os.path.join(config['data_path'], config['filename']))
    data.fillna(method='bfill', inplace=True)
    data['Patv'] = data[config['target']].apply(lambda x: max(0, x))
    logger.info(f'Raw data.shape: {data.shape}')

    # # 删除异常值， 但是删了就组不成
    # outliers1 = (data['Ndir'] > 720) | (data['Ndir'] < -720)
    # outliers2 = (data['Wdir'] > 180) | (data['Wdir'] < -180)
    # data = data[~(outliers1 | outliers2)]
    # logger.info(f'After removing outliers, data.shape: {data.shape}')

    # 按TurbID分组
    data = group_data(data, config['start_col']) # [num_nodes, seq_len, num_features]
    data = data.transpose((0, 2, 1)) # [num_nodes, num_features, seq_len]
    
    # 标准化
    means = np.mean(data, axis=(0, 2))  # [num_features, ], 每个feature的均值
    stds  = np.std(data, axis=(0, 2))  # [num_features, ],  每个feature的标准差
    data  = (data - means.reshape(1, -1, 1)) / stds.reshape(1, -1, 1)
    logger.info(f'data.shape: {data.shape}, means.shape: {means.shape}, stds.shape: {stds.shape}')


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

    # load adjency matrix
    A      = get_adjency_matrix(config['location_path'], config['thresh_distance'])
    A_wave = torch.from_numpy(get_normalized_adj(A)).to(torch.float32)

    return (train_original_data, 
            val_original_data, 
            test_original_data,
            A_wave,
            means,
            stds)


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]
    W = np.array(W)

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials