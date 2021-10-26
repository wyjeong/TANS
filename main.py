####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import os
from parser import Parser
from datetime import datetime

from misc.utils import *
from retrieval.retrieval import Retrieval

def main(args):

    set_seed(args)
    args = set_gpu(args)
    args = set_path(args)

    print(f'mode: {args.mode}')
    retrieval = Retrieval(args)
    
    if args.mode == 'train':
        # train cross-modal space from model-zoo
        retrieval.train()
    elif args.mode == 'test':
        # test cross-modal space on unseen datasets
        retrieval.test()

def set_seed(args):
    # Set the random seed for reproducible experiments
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def set_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES']= '-1' if args.gpu == None else args.gpu
    args.gpu = int(args.gpu)
    return args 

def set_path(args):

    now = datetime.now().strftime("%Y%m%d_%H%M")
    args.log_path = os.path.join(args.base_path, now, 'logs')
    args.check_pt_path = os.path.join(args.base_path, now, 'checkpoints') 

    if not os.path.exists(args.base_path):
        os.makedirs(args.base_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.check_pt_path):
        os.makedirs(args.check_pt_path)        

    if args.mode == 'train':
        args.retrieval_path = os.path.join(args.base_path, now, 'retrieval') 
        if not os.path.exists(args.retrieval_path):
            os.makedirs(args.retrieval_path)

    return args

if __name__ == '__main__':
    main(Parser().parse())
