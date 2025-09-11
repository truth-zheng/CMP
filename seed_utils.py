# seed_utils.py
import random
import numpy as np
import torch
import os

def set_all_seeds(seed=42):
    """
    设置所有随机数种子以确保实验可重复
    
    参数:
        seed (int): 随机数种子值，默认为42
    """
    # Python内置随机数生成器
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"所有随机数种子已设置为: {seed}")