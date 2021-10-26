####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import argparse
from config import *

class Parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def str2bool(self, s):
      return s.lower() in ['true', 't']
       
    def set_arguments(self):

        ##############################################
        self.parser.add_argument('--gpu', type=str, default='-1', help='gpus to use, i.e. 0')
        self.parser.add_argument('--mode', type=str, default='train', help='i.e. train, test')
        self.parser.add_argument('--seed', type=int, default=777, help='seed for reproducibility')
        self.parser.add_argument('--n-epochs', type=int, default=10000, help='number of epochs')
        self.parser.add_argument('--batch-size', type=int, default=140, help='batch size')
        self.parser.add_argument('--n-groups', type=int, default=140, help='number of meta-training datasets')
        self.parser.add_argument('--n-nets', type=int, default=100, help='number of networks per dataset')
        self.parser.add_argument('--n-dims', type=int, default=128, help='dimension of model and query embedding')
        self.parser.add_argument('--n-retrievals', type=int, default=10, help='num models to be retrieved')
        self.parser.add_argument('--n-eps-finetuning', type=int, default=50, help='num epochs for finetuning')
        self.parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    
        ##############################################
        self.parser.add_argument('--model-zoo', type=str, default='path/to/your/model/zoo', help='path to model-zoo')
        self.parser.add_argument('--model-zoo-raw', type=str, default='path/to/model/zoo/raw', help='path to raw model-zoo')
        self.parser.add_argument('--data-path', type=str, default='path/to/your/data', help='path to meta-training or meta-test dataset')
        self.parser.add_argument('--base-path', type=str, default='path/for/outcomes', help='base parent path for logging, saving, etc.')
        self.parser.add_argument('--load-path', type=str, default='path/to/outcomes/are/stored', help='base path for loading encoders, cross-modal space, etc.')
    
    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
