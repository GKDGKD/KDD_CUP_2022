import pandas as pd
from epftoolbox.evaluation import DM, plot_multivariate_DM_test
from epftoolbox.data import read_data

path_truth  = r"./result/2024_01_09_18_00_01_STGCN/ground_truths.csv"
path_rnn    = r"./result/2024_01_07_11_43_59_RNN/predictions.csv"
path_lstm   = r"./result/2024_01_07_16_49_59_LSTM/predictions.csv"
path_gru    = r"./result/2024_01_07_17_28_38_GRU/predictions.csv"
path_tcn    = r"./result/2024_01_08_16_20_51_TCN/predictions.csv"
path_stgcn  = r"./result/2024_01_07_15_52_06_STGCN/predictions.csv"
path_astgcn = r"./result/2024_01_07_11_58_54_ASTGCN/predictions.csv"
path_gtcn   = r"./result/2024_01_20_10_31_18_GTCN/predictions.csv"

print('load data...')
df_truth = pd.read_csv(path_truth)
df_rnn   = pd.read_csv(path_rnn)
df_lstm  = pd.read_csv(path_lstm)
df_gru   = pd.read_csv(path_gru)
df_tcn   = pd.read_csv(path_tcn)
df_stgcn = pd.read_csv(path_stgcn)
df_astgcn= pd.read_csv(path_astgcn)
df_gtcn  = pd.read_csv(path_gtcn)


"""
The test compares whether there is a difference in predictive accuracy between two forecast p_pred_1 and p_pred_2. 
Particularly, the one-sided DM test evaluates the null hypothesis H0 of the forecasting errors of p_pred_2 being larger (worse) than 
the forecasting errors p_pred_1 vs the alternative hypothesis H1 of the errors of p_pred_2 being smaller (better). 
Hence, rejecting H0 means that the forecast p_pred_2 is significantly more accurate that forecast p_pred_1. 
(Note that this is an informal definition. For a formal one we refer to here)
"""

print('DM test...')
norm = 1
version = 'multivariate'
# version = 'univariate'

p_rnn  = DM(p_real=df_truth.values,
     p_pred_1=df_rnn.values,
    p_pred_2=df_gtcn.values,
    norm=norm, version=version)
print(f"RNN: {p_rnn:.8f}")

p_lstm = DM(p_real=df_truth.values,
     p_pred_1=df_lstm.values,
    p_pred_2=df_gtcn.values,
    norm=norm, version=version)
print(f"LSTM: {p_lstm:.8f}")

p_gru = DM(p_real=df_truth.values,
     p_pred_1=df_gru.values,
    p_pred_2=df_gtcn.values,
    norm=norm, version=version)
print(f"GRU: {p_gru:.8f}")

p_tcn = DM(p_real=df_truth.values,
     p_pred_1=df_tcn.values,
    p_pred_2=df_gtcn.values,
    norm=norm, version=version)
print(f"TCN: {p_tcn:.8f}")

p_stgcn = DM(p_real=df_truth.values,
     p_pred_1=df_stgcn.values,
    p_pred_2=df_gtcn.values,
    norm=norm, version=version)
print(f"STGCN: {p_stgcn:.8f}")

p_astgcn = DM(p_real=df_truth.values,
     p_pred_1=df_astgcn.values,
    p_pred_2=df_gtcn.values,
    norm=norm, version=version)
print(f"ASTGCN: {p_astgcn:.8f}")

res = {
    'RNN'   : p_rnn,
    'LSTM'  : p_lstm,
    'GRU'   : p_gru,
    'TCN'   : p_tcn,
    'STGCN' : p_stgcn,
    'ASTGCN': p_astgcn
}

df_res = pd.DataFrame(res, index=['DM'])
df_res.to_csv(f'./result/DM检验结果_norm_{norm}.csv')
print('Done!')