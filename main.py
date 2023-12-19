import time, os
from log.logutli import Logger
from data_prepare import TimeSeriesDataLoader



def main():

    # Logger
    start_time   = time.time()
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time))
    save_dir     = os.path.join("result", current_time)
    log_id       = 'main'
    log_name     = f'Run_{current_time}.log'
    log_level    = 'info'
    logger       = Logger(log_id, save_dir, log_name, log_level)
    logger.logger.info(f"LOCAL TIME: {current_time}")
    
    # 结果保存路径
    save_dir_single = os.path.join(save_dir, 'Single')
    if not os.path.exists(save_dir_single):
        os.makedirs(save_dir_single)