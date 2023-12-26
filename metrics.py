import numpy as np
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                             mean_absolute_percentage_error, # 不是MAPE
                             r2_score)

def MAPE(truth, pred):
    # 范围[0,+∞)，MAPE 为0%表示完美模型，MAPE 大于 100 %则表示劣质模型。
    # 注意：当真实值有数据等于0时，存在分母0除问题，该公式不可用！
    return np.mean(np.abs((truth - pred) / truth)) * 100

# Copied from https://blog.csdn.net/guolindonggld/article/details/87856780
def SMAPE(y_true, y_pred):
    # MAPE改进，防止分母为0（SMAPE 为0%表示完美模型，SMAPE 大于 100 %则表示劣质模型。）
    # 注意：当真实值有数据等于0，而预测值也等于0时，存在分母0除问题，该公式不可用！
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def regression_metric(truth, pred):
    assert truth.shape == pred.shape, "truth.shape: {} does not match pred.shape: {}".format(truth.shape, pred.shape)
    if truth.ndim > 1 and truth.shape[1] > 1:
        truth, pred = truth.flatten(), pred.flatten()
    mse   = mean_squared_error(truth, pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(truth, pred)
    mape  = MAPE(truth, pred)
    smape = SMAPE(truth, pred)
    r2    = r2_score(truth, pred)
    score = (rmse + mae) / 2

    return {'MSE'  : mse, 
            'RMSE' : rmse,
            'MAE'  : mae,
            'MAPE' : mape,
            'SMAPE': smape,
            'R2'   : r2,
            'Score': score}

