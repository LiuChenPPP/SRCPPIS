import pickle

import numpy

from config import Config

config = Config()
device = config.device
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** (-0.5)).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_inv[np.isnan(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result
def Trans_DistanceMap(distance_map):
    result=np.where(distance_map<config.MAP_CUTOFF,1,0)
    result = result + result.T + np.eye(len(distance_map))
    return result
class PPI_Data(Dataset):
    def __init__(self, data_path, ID):
        with open(data_path,'rb') as f:
            self.all_data=pickle.load(f)
        f.close()
        self.IDProtein={}
        for key in ID:
            self.IDProtein[key]=self.all_data[key]
    def __getitem__(self, index):
        id=list(self.IDProtein.keys())[index]
        protein=self.IDProtein[id]
        AngleMap=protein['AngleMap']
        DistanceMap=protein['DistanceMap']
        DistanceMap =normalize(Trans_DistanceMap(np.transpose(DistanceMap, [2, 0, 1])[2]))
        PSSM=protein['PSSM']
        HMM=protein['HMM']
        Probert=protein['Probert']
        Label=protein['label']
        Label = numpy.array([int(l) for l in Label])
        return DistanceMap,PSSM,HMM,Probert,Label
    def __len__(self):
        return len(self.IDProtein.keys())



def propose_dataset(path):
    data = pickle.load(open(path, 'rb'))
    for key in data.keys():
        distance_map = data[key]['structure_emb']
        edge_cutoff = np.where(distance_map < config.MAX_CUTOFF, 1, 0)
        edge_cutoff = edge_cutoff + edge_cutoff.T + np.eye(len(data[key]['seq']))
        edge_KNN = np.argsort(distance_map, axis=1)[:, 0:config.MAX_K]
        mask_K = np.zeros((len(data[key]['seq']), len(data[key]['seq'])))
        for i in range(len(mask_K)):
            mask_K[i][edge_KNN[i]] = 1
        mask_K = mask_K + mask_K.T + np.eye(len(data[key]['seq']))
        edge_KNN = mask_K
        data[key]['edge_cutoff'] = edge_cutoff
        data[key]['edge_KNN'] = edge_KNN
    with open(path, 'wb+') as file:
        pickle.dump(data, file)
