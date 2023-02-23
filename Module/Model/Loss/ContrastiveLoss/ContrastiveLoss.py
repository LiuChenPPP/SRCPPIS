import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import seaborn as sns
import random
def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception
class Contrastiveloss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(Contrastiveloss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # euclidean_distance = F.pairwise_distance(output1, output2)
        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))

        return loss_contrastive