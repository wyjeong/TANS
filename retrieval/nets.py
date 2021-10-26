####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import math
import torch
import numpy as np
from config import *
from misc.utils import *
import torch.nn.functional as F

class PerformancePredictor(torch.nn.Module):
    def __init__(self, args):
        super(PerformancePredictor, self).__init__()
        self.args = args
        self.fc = torch.nn.Linear(self.args.n_dims*2, 1)

    def forward(self, q, m):
        p = torch.cat([q, m], 1)
        p = torch.sigmoid(self.fc(p))
        return p

class ModelEncoder(torch.nn.Module):
    def __init__(self, args):
        super(ModelEncoder, self).__init__()  
        self.args = args
        self.fc = torch.nn.Linear(45+1536, self.args.n_dims) 
            
    def forward(self, v_t, v_f):
        m = torch.cat([v_t, v_f], 1)
        m = F.normalize(m)
        m = self.fc(m)
        m = self.l2norm(m)
        return m

    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x
    
class QueryEncoder(torch.nn.Module):
    def __init__(self, args):
        super(QueryEncoder, self).__init__()
        self.args = args
        self.fc = torch.nn.Linear(512, self.args.n_dims) 

    def forward(self, D):
        q = []
        for d in D:
            _q = self.fc(d) 
            _q = torch.mean(_q, 0)
            _q = self.l2norm(_q.unsqueeze(0))
            q.append(_q)
        q = torch.stack(q).squeeze()
        return q

    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x

