# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:00:49 2023

@author: User
"""

import rdkit
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')        
from rdkit.Chem import MolFromSmiles,MolToSmiles
from tqdm import tqdm
import numpy as np

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

def LinearAnnealing(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

class MolData(Dataset):
    def __init__(self,smis,props,toks):
        self.smis = smis
        self.props = props
        self.toks = toks + ['<','>'] #'<'; start of sequence, '>'; end of sequence
        self.Ntok = len(toks)
        self.Nmax = 120
        
    def __len__(self):
        return len(self.smis)

    def __getitem__(self,idx):
        smi = '<'+self.smis[idx]+'>'
        smi += '>'*(self.Nmax-len(smi))

        x_all = np.array([self.toks.index(s) for s in smi]).flatten()

        #x = x_all[:-1] #input
        y = x_all[1:] #output
        #prop = logP(MolFromSmiles(self.smis[idx]))
        #prop = penalized_logP(MolFromSmiles(self.smis[idx]))

        x = torch.LongTensor(x_all)
        y = torch.LongTensor(y)
        prop = torch.Tensor([self.props[idx]])
        return x,y,prop

class MolVAE(nn.Module):
    def __init__(self,dim_x0,dim_x1,dim_h,n_layer,d_ratio,dim_z,dim_zh,d_ratio2):
        super(MolVAE,self).__init__()
        self.n_layer = n_layer
        self.emb_layer = nn.Embedding(dim_x0,dim_x1)
        
        self.enc = nn.GRU(dim_x1,dim_h,
                          num_layers=n_layer,
                          dropout = d_ratio,
                          batch_first = True)
        
        self.fc_z1 = nn.Sequential(
                        nn.Linear(dim_h,dim_h),
                        nn.ReLU(),
                        nn.Linear(dim_h,2*dim_z))

        self.z_to_prop = nn.Sequential(
                        nn.Linear(dim_z,dim_zh),
                        nn.ReLU(),
                        nn.Dropout(p=d_ratio2),
                        nn.Linear(dim_zh,dim_zh),
                        nn.ReLU(),
                        nn.Dropout(p=d_ratio2),
                        nn.Linear(dim_zh,dim_zh),
                        nn.ReLU(),
                        nn.Linear(dim_zh,1))
    
        self.fc_z2 = nn.Linear(dim_z,dim_h)
        
        self.dec = nn.GRU(dim_x1+dim_z,dim_h,
                          num_layers=n_layer,
                          dropout = d_ratio,
                          batch_first = True)

        self.out = nn.Sequential(
                        nn.Linear(dim_h,dim_h),
                        nn.ReLU(),
                        nn.Linear(dim_h,dim_x0))
        
    def forward(self,x):
        x_emb = self.emb_layer(x)
        
        mu,log_var = self.encoder(x_emb)
        eps = torch.randn_like(mu).cuda()
        z = mu + eps*torch.exp(log_var/2)
        
        out = self.decoder(x_emb[:,:-1],z)
        p_out = self.regressor(z)
        return out,p_out,mu,log_var
       
    def encoder(self,x):
        _,h1 = self.enc(x,None)
        h2 = self.fc_z1(h1[-1])
        mu,log_var = torch.chunk(h2,2,dim=-1)
        return mu,log_var
        
    def decoder(self,x,z):
        N,L,F = x.shape
        h0_z = z.unsqueeze(1).repeat(1,L,1)
        
        x_in = torch.cat([x,h0_z],dim=-1)
        
        h0_rnn = self.fc_z2(z).unsqueeze(0).repeat(self.n_layer,1,1)
        out,h_d = self.dec(x_in,h0_rnn)
        out = self.out(out)
        return out

    def regressor(self,z):
        prop = self.z_to_prop(z)
        return prop

    def sampling(self,x0,z,h0=None,is_first=True):
        x = self.emb_layer(x0)

        N,L,F = x.shape
        h0_z = z.unsqueeze(1).repeat(1,L,1)
        x_in = torch.cat([x,h0_z],dim=-1)

        if is_first:
            h0 = self.fc_z2(z).unsqueeze(0).repeat(self.n_layer,1,1)
        
        out,h1 = self.dec(x_in,h0)
        out = self.out(out)
        return out,h1
        

def Sampling(sampler,dim_z,n_sample,max_len,tok_lib):
    sampler.eval()
    with torch.no_grad():
        inits = torch.LongTensor([34]*n_sample)
        loader = DataLoader(inits,batch_size=100)

        Sampled = []
        Zs = []
        Ps = []
        for inp in tqdm(loader):
            x_in = inp.cuda().reshape(-1,1)
        
            x_hat = []
            z = torch.randn(len(x_in),dim_z).cuda()
            pp = sampler.regressor(z)
            Ps.append(pp.cpu().detach().numpy().reshape(-1,1))

            h = None
            is_first = True
            for seq_iter in range(max_len):

                if seq_iter > 0:
                    is_first = False

                out,h = sampler.sampling(x_in,z,h,is_first)
                prob = F.softmax(out,dim=-1).squeeze(1)
                x_in = torch.multinomial(prob,1)
                #x_in = torch.argmax(prob,-1).view(-1,1)
                
                x_hat.append(x_in.cpu().detach().numpy())

            x_hat = np.hstack(x_hat)
            Sampled.append(x_hat)
            Zs.append(z.cpu().detach().numpy())
            
    Sampled = np.vstack(Sampled)
    Zs = np.vstack(Zs)
    Ps = np.vstack(Ps).flatten()
    
    Mols = []
    Lat1 = []
    P1 = []
    for s,z,p in zip(Sampled,Zs,Ps):
        n_end = np.sum(s==35)
        
        if n_end == 0:
            continue
        
        n = np.min(np.where(s==35)[0])
        m = ''.join(tok_lib[s[:n]].tolist())
        Mols.append(m)
        Lat1.append(z)
        P1.append(p)
        
    Vals = []
    Lat2 = []
    P2 = []
    for smi,z,p in zip(Mols,Lat1,P1):
        mol = MolFromSmiles(smi)
        if not mol is None:
            Vals.append(MolToSmiles(mol))
            Lat2.append(z.reshape(1,-1))
            P2.append(p)

    Uni = list(set(Vals))
    Lat2 = np.vstack(Lat2)
    P2 = np.array(P2)
    return Vals,Lat2,P2,len(Vals),len(Uni)
