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

from rnn_funcs import MolData,MolVAE,Sampling,LinearAnnealing

toks = pickle.load(open('ZINC_Tokens.pkl','rb'))

smi_ = [ss.split()[0] for ss in pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv')['smiles']]

np.random.seed(1)
np.random.shuffle(smi_)

n_train = 200000
n_val = 25000
n_t = 25000

smi_train = smi_[:n_train]
smi_val = smi_[n_train:n_train+n_val]
#smi_test = smi_[n_train+n_val:]

#smi_test = list(pd.read_csv('test.txt')['SMILES'])

batch_size = 64

train_data = MolData(smi_train,toks)
tok_lib = np.array(train_data.toks) # For sampling
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

test_data = MolData(smi_val,toks)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

DimZ = 156
model = MolVAE(36,128,480,3,0.2,DimZ).cuda()

lr = 1e-4
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

num_epoch = 200
max_norm = 5

LOGs = []
Betas = LinearAnnealing(n_iter=num_epoch,start=0.0,stop=0.2).tolist()

for ep in range(num_epoch):
    model.train()
    for inp in tqdm(train_loader):
        x_in = inp[0].cuda()
        tgt = inp[1].cuda().view(-1)
        
        x_out,mu,log_var = model(x_in)
        
        rec = ce_loss(x_out.reshape(-1,36),tgt)
        kld = torch.mean(0.5*(mu**2+torch.exp(log_var)-log_var-1))
        
        loss = rec + Betas[ep]*kld
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        Sim = []
        Mus = []
        Stds = []
        KLDs = 0
        Ns = 0
        for inp in tqdm(test_loader):
            x_in = inp[0].cuda()
            tgt = inp[1].cuda().view(-1)
            
            x_out,mu,log_var = model(x_in)

            kld = torch.sum(torch.mean(0.5*(mu**2+torch.exp(log_var)-log_var-1),-1))

            KLDs += kld.cpu().detach().numpy().flatten()[0]
            Ns += len(x_in)
            
            id_out = np.argmax(x_out.cpu().detach().numpy(),-1)
            id_in = x_in[:,1:].cpu().detach().numpy()
            acc = np.mean(id_out==id_in,1).reshape(-1,1)
            
            Sim.append(acc)
            Mus.append(mu.cpu().detach().numpy())
            Stds.append(torch.exp(log_var/2).cpu().detach().numpy())
            
        Sim = np.vstack(Sim)
        Mus = np.vstack(Mus)
        Stds = np.vstack(Stds)
        mols,Zs,val,uniq = Sampling(model,DimZ,10000,120,tok_lib) 
            
        print(ep,Betas[ep],np.min(Sim),np.max(Sim),np.mean(Sim),np.std(Sim),KLDs/Ns,val,uniq)
        LOGs.append([ep,Betas[ep],np.min(Sim),np.max(Sim),np.mean(Sim),np.std(Sim),KLDs/Ns,val,uniq])
       
        res = {'state_dict':model.state_dict(),'Mus':Mus,'Vars':Stds,'Mols':mols,'Zs':Zs,'Val':val,'Unique':uniq}
        torch.save(res,'MolVAE/Logs_'+str(ep)+'.pth.tar')

pickle.dump(np.array(LOGs),open('MolVAE/LearningLog.pkl','wb'))
