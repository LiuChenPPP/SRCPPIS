import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch
import math


class MHAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.nhead = n_heads

        self.dropout = nn.Dropout(dropout)
        self.hidden_size_head = int(self.hid_dim / self.nhead)

    def forward(self, v, k, q, mask):
        torch.cuda.empty_cache()
        n_batches = q.size(0)
        # torch.cuda.memory_summary(device=None, abbreviated=False)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hid_dim
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)




class crossAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(crossAttention, self).__init__()

        self.mhatt1 = MHAtt(hid_dim, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hid_dim)

    def forward(self, x, y, y_mask=None):
        torch.cuda.empty_cache()

        # x as V while y as Q and K
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, y, y_mask)
        # ))
        x = self.norm1(x + self.dropout1(
            self.mhatt1(y, y, x, y_mask)
        ))


        return x


class SequenceGraphAttention(nn.Module):
    def __init__(self, hid_dim=256, n_heads=8, dropout=0.3):
        super(SequenceGraphAttention, self).__init__()

        self.coa_pc = crossAttention(hid_dim, n_heads, dropout)
        self.coa_cp = crossAttention(hid_dim, n_heads, dropout)

    def forward(self, seuqnece_vector, graph_vector):
        seuqnece_vector = self.coa_pc(seuqnece_vector, graph_vector, None)  # co-attention
        graph_vector = self.coa_cp(graph_vector, seuqnece_vector, None)  # co-attention

        return seuqnece_vector, graph_vector