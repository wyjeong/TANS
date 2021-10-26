####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import torch
import torch.nn as nn

class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2, contrast=False):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax
        self.contrast = contrast

    def forward(self, m, q, matched=None):
        scores = torch.mm(m, q.t()) # (160, 160)
        diag = scores.diag() # (160,)
        scores = (scores - 1 * torch.diag(scores.diag())) 
        # Sort the score matrix in the query dimension
        sorted_query, _ = torch.sort(scores, 0, descending=True)
        # Sort the score matrix in the model dimension
        sorted_model, _ = torch.sort(scores, 1, descending=True)
        # Select the nmax score
        max_q = sorted_query[:self.nmax, :] # (1, 160)
        max_m = sorted_model[:, :self.nmax] # (160, 1)
        neg_q = torch.sum(torch.clamp(max_q + 
            (self.margin - diag).view(1, -1).expand_as(max_q), min=0))
        neg_m = torch.sum(torch.clamp(max_m + 
            (self.margin - diag).view(-1, 1).expand_as(max_m), min=0))

        if self.contrast:
            loss = neg_m + neg_q
        else:
            loss = neg_m

        return loss