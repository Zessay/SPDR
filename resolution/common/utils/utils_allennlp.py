# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-07
import os
import random
import torch
import logging
import numpy as np
from pathlib import Path


def seed_everything(seed=2020):
    """设置整个开发环境的seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def init_logger(log_file=None, log_file_level=logging.NOTSET, log_on_console=True):
    """用于初始化log对象，可以选择是否记录在文件中以及是否在屏幕上显示"""
    if isinstance(log_file, Path):
        log_file = str(log_file)

    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    if log_on_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.handlers = [console_handler]
    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger