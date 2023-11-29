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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

class MolData(Dataset):
    def __init__(self,smis,toks):
        self.smis = smis
        self.toks = toks + ['<','>'] #'<'; start of sequence, '>'; end of sequence
        self.Ntok = len(toks)
        self.Nmax = 60
        
    def __len__(self):
        return len(self.smis)
    
    def __getitem__(self,idx):
        smi = '<'+self.smis[idx]+'>'
        smi += '>'*(self.Nmax-len(smi))

        x_all = np.array([self.toks.index(s) for s in smi]).flatten()

        x = x_all[:-1] #input
        y = x_all[1:] #output

        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return x,y
    
class CharRNN(nn.Module):
    def __init__(self,dim_x0,dim_x1,dim_h,n_layer,d_ratio):
        super(CharRNN,self).__init__()
        self.n_layer = n_layer
        self.emb_layer = nn.Embedding(dim_x0,dim_x1)
        
        self.rnn = nn.GRU(dim_x1,dim_h,
                          num_layers=n_layer,
                          dropout = d_ratio,
                          batch_first = True)

        self.out = nn.Sequential(
                        nn.Linear(dim_h,dim_h),
                        nn.ReLU(),
                        nn.Linear(dim_h,dim_x0))
        
    def forward(self,x_in,h0=None):
        x0 = self.emb_layer(x_in)
        x1,h1 = self.rnn(x0,h0)
        x2 = self.out(x1)
        return x2,h1
        
def Sampling(sampler,n_sample,max_len,tok_lib):
    sampler.eval()
    with torch.no_grad():
        inits = torch.LongTensor([26]*n_sample)
        loader = DataLoader(inits,batch_size=100)

        Sampled = []
        for inp in tqdm(loader):
            x_in = inp.cuda().reshape(-1,1)
        
            x_hat = []
            h = None
            for seq_iter in range(max_len):
                out,h = sampler(x_in,h)
                prob = F.softmax(out,dim=-1).squeeze(1)
                x_in = torch.multinomial(prob,1)
                #x_in = torch.argmax(prob,1).view(-1,1)
                
                x_hat.append(x_in.cpu().detach().numpy())

            x_hat = np.hstack(x_hat)
            Sampled.append(x_hat)
            
    Sampled = np.vstack(Sampled)
    
    Mols = []
    for s in Sampled:
        n_end = np.sum(s==27)
        
        if n_end == 0:
            continue
        
        n = np.min(np.where(s==27)[0])
        m = ''.join(tok_lib[s[:n]].tolist())
        Mols.append(m)
        
    Vals = []
    for smi in Mols:
        mol = MolFromSmiles(smi)
        if not mol is None:
            Vals.append(MolToSmiles(mol))
            
    Uni = list(set(Vals))
    
    return Vals,len(Vals),len(Uni)
