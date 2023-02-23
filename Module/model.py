import numpy as np
from config import Config

config = Config()
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from Module.Model.MASK.MASK import Mask_Attention
from Model.MASK.MASK import Mask_Attention
from Model.GTM.GTM import GTM
from Model.CrossAttention.CrossAttention import SequenceGraphAttention


class GAT(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_feature, out_feature, dropout=dropout, concat=True) for _
                           in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(out_feature * nheads, out_feature, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, residual=None, concat=True, ):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(config.leakRl_Alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCN_GAT(nn.Module):
    def __init__(self, infeature, outfeature, lamda, alpha,
                 dropout, heads):
        super(GCN_GAT, self).__init__()
        self.lamda = lamda
        self.alpha = alpha
        self.gat = GAT(infeature, outfeature, dropout, heads)
        self.gcn = GraphConvolution(infeature, outfeature, residual=False)
        self.ln1 = nn.LayerNorm(infeature)
        self.ln2 = nn.LayerNorm(infeature)
        self.relu = nn.LeakyReLU(config.leakRl_Alpha)
        self.cat_lin = nn.Linear(infeature * 2, outfeature)

    # layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.alpha, i + 1))
    def forward(self, x, adj, h0, l):
        input = x
        x1 = self.gat(x, adj)
        x2 = self.gcn(x, adj, h0, self.lamda, self.alpha, l)
        x = torch.cat((x1, x2), dim=1)
        x = self.cat_lin(x)
        return x + input


class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant, heads):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.ln = nn.LayerNorm(nhidden)
        for _ in range(nlayers):
            # self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
            self.convs.append(
                GCN_GAT(infeature=nhidden, outfeature=nhidden, lamda=lamda, alpha=alpha, dropout=dropout, heads=heads))
            # self.convs.append(GraphAttentionLayer(nhidden,nhidden,dropout=0.3))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.LeakyReLU(config.leakRl_Alpha)
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.augment_eps = 0.05
        self.lin1 = nn.Linear(54, 256)
        self.lin2 = nn.Linear(1024, 256)
        self.help_lin = nn.Linear(256, 256)

    def forward(self, x, adj, seq_emb):
        # esm_1b = self.esm_1b_liner(esm_1b)
        # x = self.lin1(x)
        # seq_emb = self.lin2(seq_emb)
        # x = torch.cat((x, seq_emb), dim=1)

        _layers = []
        # x = F.dropout(x, self.dropout, training=self.training)
        # layer_inner = self.act_fn(self.fcs[0](x))
        layer_inner = x
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(self.ln(con(layer_inner, adj, _layers[0], i + 1)))
        layer_inner = self.help_lin(layer_inner)
        return layer_inner


class PPIGraph(nn.Module):
    def __init__(self):
        super(PPIGraph, self).__init__()
        self.graph_extract1 = deepGCN(config.LAYER, config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_CLASSES,
                                      config.DROPOUT, config.LAMBDA, config.ALPHA, config.VARIANT, config.heads)
        self.graph_extract2 = deepGCN(config.LAYER, config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_CLASSES,
                                      config.DROPOUT, config.LAMBDA, config.ALPHA, config.VARIANT, config.heads)
        self.augment_eps = 0.05
        self.GTM = GTM(256)
        self.result = nn.Linear(256, 2)
        self.mask_distance = Mask_Attention(256, 128, 128, 128, 2, 1)
        self.lin1 = nn.Linear(54, 256)
        self.lin2 = nn.Linear(1024, 256)
        self.lin3 = nn.Linear(1280, 256)
        self.lin = nn.Linear(256 * 3, 256)
        self.out = nn.Linear(512, 256)

        self.CrossAttention = nn.ModuleList([SequenceGraphAttention() for _ in range(config.LAYER)])

        self.input_block_seq_emb = nn.Sequential(
            nn.LayerNorm(1024, eps=1e-6)
            , nn.Linear(1024, config.HIDDEN_DIM)
            , nn.LeakyReLU(config.leakRl_Alpha)
        )

        self.hidden_block_seq_emb = nn.Sequential(
            nn.LayerNorm(config.HIDDEN_DIM, eps=1e-6)
            , nn.Dropout(config.DROPOUT)
            , nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
            , nn.LeakyReLU(config.leakRl_Alpha)
            , nn.LayerNorm(config.HIDDEN_DIM, eps=1e-6)
        )

        self.input_block_pssm = nn.Sequential(
            nn.LayerNorm(20, eps=1e-6)
            , nn.Linear(20, config.HIDDEN_DIM)
            , nn.LeakyReLU(config.leakRl_Alpha)
        )

        self.hidden_block_pssm = nn.Sequential(
            nn.LayerNorm(config.HIDDEN_DIM, eps=1e-6)
            , nn.Dropout(config.DROPOUT)
            , nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
            , nn.LeakyReLU(config.leakRl_Alpha)
            , nn.LayerNorm(config.HIDDEN_DIM, eps=1e-6)
        )
        self.input_block_dssp = nn.Sequential(
            nn.LayerNorm(14, eps=1e-6)
            , nn.Linear(14, config.HIDDEN_DIM)
            , nn.LeakyReLU(config.leakRl_Alpha)
        )

        self.hidden_block_dssp = nn.Sequential(
            nn.LayerNorm(config.HIDDEN_DIM, eps=1e-6)
            , nn.Dropout(config.DROPOUT)
            , nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
            , nn.LeakyReLU(config.leakRl_Alpha)
            , nn.LayerNorm(config.HIDDEN_DIM, eps=1e-6)
        )
        self.input_block_hhm = nn.Sequential(
            nn.LayerNorm(20, eps=1e-6)
            , nn.Linear(20, config.HIDDEN_DIM)
            , nn.LeakyReLU(config.leakRl_Alpha)
        )

        self.hidden_block_hhm = nn.Sequential(
            nn.LayerNorm(config.HIDDEN_DIM, eps=1e-6)
            , nn.Dropout(config.DROPOUT)
            , nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
            , nn.LeakyReLU(config.leakRl_Alpha)
            , nn.LayerNorm(config.HIDDEN_DIM, eps=1e-6)
        )

        self.block1 = nn.Sequential(
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(config.leakRl_Alpha),
            nn.Linear(512, 256),
        )

        self.block2 = nn.Sequential(
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(config.leakRl_Alpha),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(config.leakRl_Alpha),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(config.leakRl_Alpha),
            nn.Linear(64, 2)
        )

    def forward(self, pssm, hhm, adj1, seq_emb):
        # esm1b=self.lin3(esm1b)
        pssm = self.hidden_block_pssm(self.input_block_pssm(pssm))
        hhm = self.hidden_block_hhm(self.input_block_hhm(hhm))
        if self.training:
            seq_emb = seq_emb + self.augment_eps * torch.randn_like(seq_emb)
        seq_emb = self.hidden_block_seq_emb(self.input_block_seq_emb(seq_emb))
        x = torch.cat((pssm, hhm, seq_emb), dim=1)
        x = self.lin(x)
        graph1 = self.graph_extract1(x, adj1, x)
        mask_ = torch.where(adj1 == 0, True, False)
        mask_ = self.mask_distance(x.unsqueeze(dim=0), mask_.unsqueeze(dim=0)).squeeze(dim=0)
        for i in range(config.LAYER):
            sequence_vector, graph_vector = self.CrossAttention[i](mask_, graph1)
        sequence_vector = sequence_vector.mean(dim=1)
        graph_vector = graph_vector.mean(dim=1)


        result = torch.cat((sequence_vector, graph_vector), dim=1)
        result = self.block1(result)
        result = self.block2(result)
        return result
