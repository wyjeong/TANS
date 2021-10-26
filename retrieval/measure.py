####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import torch
import numpy as np


def compute_recall(query_embs, model_embs, npts=None):
    if npts is None:
        npts = query_embs.shape[0] 
    index_list = []
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        # Get query_embs image
        query_emb = query_embs[index].reshape(1, query_embs.shape[1])
        # Compute scores
        d = np.dot(query_emb, model_embs.T).flatten()
        sorted_index_lst = np.argsort(d)[::-1]
        index_list.append(sorted_index_lst[0])
        # Score
        rank = np.where(sorted_index_lst == index)[0][0]
        ranks[index] = rank
        top1[index] = sorted_index_lst[0]
    recalls = {}
    for v in [1, 5, 10, 50, 100]:
        recalls[v] = 100.0 * len(np.where(ranks < v)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return recalls, medr, meanr

