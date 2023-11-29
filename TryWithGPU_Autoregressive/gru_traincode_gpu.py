# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:33:01 2023

@author: User
"""

import numpy as np
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 0 to use

import pickle

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from gru_funcs import MolData,CharRNN,Sampling

toks = pickle.load(open('MOSES_Tokens.pkl','rb'))

smi_ = list(pd.read_csv('train.txt')['SMILES'])

np.random.seed(1)
np.random.shuffle(smi_)

n_train = 1000000
n_val = 100000

smi_train = smi_[:n_train]
smi_val = smi_[n_train:n_train+n_val]

#smi_test = list(pd.read_csv('test.txt')['SMILES'])

batch_size = 64

train_data = MolData(smi_train,toks)
tok_lib = np.array(train_data.toks) # For sampling
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

model = CharRNN(28,128,480,3,0.2).cuda()

lr = 2e-4
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

num_epoch = 20
max_norm = 5
num_iter = 0

os.makedirs('CharGRU',exist_ok=True)

save_id = 0
model.train()
for ep in range(num_epoch):
    for inp in tqdm(train_loader):
        x_in = inp[0].cuda()
        tgt = inp[1].cuda()
        
        x_out,_ = model(x_in)
       
        loss = ce_loss(x_out.reshape(-1,28),tgt.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        num_iter += 1

        if num_iter % 5000 == 0:
            mols,val,uniq = Sampling(model,10000,100,tok_lib)
            print(ep,num_iter,val,uniq)
            res = {'state_dict':model.state_dict(),'Mols':mols,'Val':val,'Unique':uniq}
            torch.save(res,'CharGRU/Logs_'+str(save_id)+'.pth.tar')
            save_id += 1

            model.train()

