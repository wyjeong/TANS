####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import os
import glob
import torch
import time
import random
import numpy as np
import torchvision
from torchvision import transforms 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from misc.utils import *
        
def get_loader(args, mode='train'):
    dataset = MetaTrainDataset(args, mode=mode)
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=(mode=='train'),
                        num_workers=4)
    return dataset, loader

def get_meta_test_loader(args, mode='train'):
    dataset = MetaTestDataset(args, mode=mode)
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=(mode=='train'),
                        num_workers=4)
    return dataset, loader

class MetaTestDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.dataset_list = [
                'recognizance1_0_5',
                'gemstones-images_lsind18_18_36',
                'drr-sign_0_4',
                'ucfai-core-fa19-cnns_76_95',
                'honey-bee-pollen_ivanfel_0_2',
                'alien-vs-predator-images_pmigdal_0_2',
                'kuzushiji_anokas_1120_1140',
                'covid19-radiography-database_tawsifurrahman_0_3',
                'colorectal-histology-mnist_kmader_0_8',
                'ml2020spring-hw12_0_10',
        ]
        self.data = torch_load(os.path.join(self.args.data_path, f'meta_test_{self.dataset_list[0]}.pt'))
        self.curr_dataset = self.dataset_list[0]

    def set_mode(self, mode):
        self.mode = mode

    def get_dataset_list(self):
        return self.dataset_list

    def set_dataset(self, dataset):
        self.curr_dataset = dataset
        self.data = torch_load(os.path.join(self.args.data_path, f'meta_test_{dataset}.pt'))
        print(f"{dataset}: #_train: {len(self.data['x_train'])}, #_test: {len(self.data['x_test'])}")

    def __len__(self):
        return len(self.data[f'x_{self.mode}'])

    def __getitem__(self, index):
        x = self.data[f'x_{self.mode}'][index]
        y = self.data[f'y_{self.mode}'][index]
        return x, y

    def get_query_set(self, task):
        return self.data[f'query']

    def get_n_clss(self):
        return self.data['nclss']

class MetaTrainDataset(Dataset):

    def __init__(self, args, mode='train'):
        start_time = time.time()
        self.args = args
        self.mode = mode
        self.model_zoo = torch_load(self.args.model_zoo)
        self.query = torch_load(os.path.join(self.args.data_path, 'meta_train.pt'))
        start_time = time.time()
        self.contents = []
        self.dataset_list = set(self.model_zoo['dataset'])
        for dataset in self.dataset_list:
            models = []
            cnt = 0
            for idx, _dataset in enumerate(self.model_zoo['dataset']):
                if dataset == _dataset:
                    cnt+= 1
                    if cnt <= self.args.n_nets:
                        ############################################
                        topol = self.model_zoo['topol'][idx]
                        ks = topol[:20] 
                        e = topol[20:40]
                        d = topol[40:]
                        tmp = torch.zeros(len(ks))
                        for stage, num_layer in enumerate(d):
                            tmp[stage*4:stage*4+num_layer] = 1
                        ks = torch.tensor(ks) * tmp
                        e = torch.tensor(e) * tmp
                        topol = [int(t) for t in [*ks.tolist(), *e.tolist(), *d]]
                        ############################################
                        models.append({
                            'acc': self.model_zoo['acc'][idx],
                            'topol': self.model_zoo['topol'][idx],
                            'f_emb': self.model_zoo['f_emb'][idx],
                            'n_params': self.model_zoo['n_params'][idx],
                        })
            self.contents.append((dataset, models))
        print(f"{len(self.contents)*self.args.n_nets} pairs loaded ({time.time()-start_time:.3f}s) ")

    def __len__(self):
        return len(self.contents) 

    def set_mode(self, mode):
        self.mode = mode
        
    def __getitem__(self, index):
        dataset = self.contents[index][0]
        n_models = len(self.contents[index][1])
        if n_models == 1:
            idx = 0 
        else:
            idx = random.randint(0,n_models-1)
        model = self.contents[index][1][idx]
        acc = model['acc'] 
        n_params = model['n_params']
        topol = torch.Tensor(model['topol'])
        f_emb = model['f_emb']
        return dataset, acc, topol, f_emb

    def get_query(self, datasets):
        x_batch = []
        for d in datasets:
            x = self.query[d][f'x_query_{self.mode}']
            x_batch.append(torch.stack(x))
        return x_batch





    