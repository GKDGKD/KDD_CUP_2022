import numpy as np
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                             mean_absolute_percentage_error, 
                             r2_score)

def regression_metric(truth, pred):
    assert truth.shape == pred.shape, "truth.shape: {} does not match pred.shape: {}".format(truth.shape, pred.shape)
    if truth.ndim > 1 and truth.shape[1] > 1:
        truth, pred = truth.flatten(), pred.flatten()
    mse   = mean_squared_error(truth, pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(truth, pred)
    mape  = mean_absolute_percentage_error(truth, pred)
    r2    = r2_score(truth, pred)
    score = (rmse + mae) / 2

    return {'MSE'  : mse, 
            'RMSE' : rmse,
            'MAE'  : mae,
            'MAPE' : mape,
            'R2'   : r2,
            'Score': score}

