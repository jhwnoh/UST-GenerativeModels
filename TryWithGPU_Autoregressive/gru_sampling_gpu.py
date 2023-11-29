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

import sys
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
smi_train = smi_[:100]

train_data = MolData(smi_train,toks)
tok_lib = np.array(train_data.toks) # For sampling

model = CharRNN(28,128,480,3,0.2).cuda()
chkpt_name = sys.argv[1]
chkpt = torch.load(chkpt_name)['state_dict']
model.load_state_dict(chkpt)

mols,val,uniq = Sampling(model,10000,100,tok_lib)

save_name = sys.argv[2]
pickle.dump([mols,val,uniq],open(save_name,'wb'))
