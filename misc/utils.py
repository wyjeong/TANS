####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import os
import pdb
import json
import torch
import random
import numpy as np
from datetime import datetime

def get_device(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def torch_save(base_dir, filename, data):
    if os.path.isdir(base_dir) == False:
        os.makedirs(base_dir)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)
    print('file saved ({})'.format(fpath))

def torch_load(fpath):
    return torch.load(fpath, map_location=torch.device('cpu'))

def f_write(filepath, filename, data):
    if os.path.isdir(filepath) == False:
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'w+') as outfile:
        json.dump(data, outfile)

def random_shuffle(seed, _list):
    random.seed(seed)
    random.shuffle(_list)

def random_sample(seed, _list, num_pick):
    random.seed(seed)
    return random.sample(_list, num_pick)

def random_int(seed, start, end):
    random.seed(seed)
    random.randint(start, end)

def shuffle(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.arange(len(x))
    random_shuffle(SEED, idx)
    return x[idx], y[idx]

def debugger():
    pdb.set_trace()

