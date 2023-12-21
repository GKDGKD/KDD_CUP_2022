import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def regression_metric(truth, pred):
    mse  = mean_squared_error(truth, pred)
    mae  = mean_absolute_error(truth, pred)
    mape = mean_absolute_percentage_error(truth, pred)
    r2   = r2_score(truth, pred)

    return {'mse': mse, 'mae': mae, 'mape': mape, 'r2': r2}

