import os
import random
import torch
import numpy as np
import json
import pickle
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
import logging

logger = logging.getLogger()


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {str(v)}\n"
    print("\n" + info + "\n")
    return


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, file_path):
    '''
    保存成json文件
    :param data:
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def save_numpy(data, file_path):
    '''
    保存成.npy文件
    :param data:
    :param file_path:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.save(str(file_path),data)


def load_numpy(file_path):
    '''
    加载.npy文件
    :param file_path:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.load(str(file_path))


def load_json(file_path):
    '''
    加载json文件
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


def json_to_text(file_path,data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')

