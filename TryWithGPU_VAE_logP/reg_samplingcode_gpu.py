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
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from rnn_reg_funcs import MolData,MolVAE,LinearAnnealing,Sampling

import rdkit
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')        
from rdkit.Chem import MolFromSmiles,MolToSmiles

toks = pickle.load(open('ZINC_Tokens.pkl','rb'))

smi_ = [ss.split()[0] for ss in pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv')['smiles']]
prop_ = np.array(pd.read_csv('250k_rndm_zinc_drugs_clean_3.csv')['logP'])

iis = np.arange(len(smi_))
np.random.seed(1)
np.random.shuffle(iis)

n_train = 200000
n_val = 25000
n_t = 25000

smi_train = [smi_[i] for i in iis[:n_train]]

pp_train = [prop_[i] for i in iis[:n_train]]

#smi_test = list(pd.read_csv('test.txt')['SMILES'])

batch_size = 64

train_data = MolData(smi_train,pp_train,toks)
tok_lib = np.array(train_data.toks) # For sampling
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

DimZ = 156
model = MolVAE(36,128,480,3,0.2,DimZ,64,0.15).cuda()

chkpt_name = sys.argv[1]
chkpt = torch.load(chkpt_name)
model.load_state_dict(chkpt['state_dict'])

mols,Zs,Ps,val,uniq = Sampling(model,DimZ,10000,120,tok_lib)
print(val,uniq)
savename = sys.argv[2]
pickle.dump([mols,Zs,Ps,val,uniq],open(savename,'wb'))
