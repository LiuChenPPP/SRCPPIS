import torch.nn as nn
import torch
import datetime, time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
class Config():
    def __init__(self):
        self.epochs =50
        self.lr = 0.01
        self.auprc = 0
        self.MAX_K=10
        self.SocialData='../dataset/Social/SocialProtein.pkl'
        self.SocialDataID = '../dataset/Social/ID.txt'
        self.data_732 = '../dataset/737_72_164_186_315/Dset732.pkl'
        self.data_732_train = '../dataset/737_72_164_186_315/TrainId.txt'
        self.data_732_test = '../dataset/737_72_164_186_315/TestId.txt'
        self.data_1082 = '../dataset/1082_448_634/Dset1082.pkl'
        self.data_1082_train = '../dataset/1082_448_634/TrainId.txt'
        self.data_1082_test = '../dataset/1082_448_634/TestId.txt'
        self.Independent='../dataset/IndependentData/Dset88.pkl'
        self.Independent_test='../dataset/IndependentData/ID.txt'

        self.seed = 10
        self.MAP_CUTOFF = 14
        self.HIDDEN_DIM = 256
        self.LAYER = 2
        self.DROPOUT = 0.4
        self.leakRl_Alpha=0.2
        self.weight_decay = 1e-4
        self.ALPHA = 0.1
        self.LAMBDA = 0.5
        self.VARIANT = True  # From GCNII
        self.eta_min = 1e-5
        self.WEIGHT_DECAY = 0
        self.BATCH_SIZE = 1
        self.NUM_CLASSES = 2  # [not bind, bind]
        self.INPUT_DIM = 256 * 2
        self.split_rate = 0.2
        self.batch_size = 1
        self.min_lr = 1e-5
        self.heads = 1
        self.save_path = 'result/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_fun = nn.CrossEntropyLoss().to(self.device)
        self.Threashold = 0.18
        # 格式化时间字符串
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S')
        with open('time.txt','a') as file:
            file.write(self.time+'\n')

        for name, value in vars(self).items():
            print(name, value)
