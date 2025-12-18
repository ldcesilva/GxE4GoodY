root_path = "//tdl/Public2/G2F/Manuscript/" # Adjust to your path
prep_path = root_path + "/Results/DataPrep/"
train_path = root_path + "/Data/Challenge2024/Training_data/"
test_path = root_path + "/Data/Challenge2024/Testing_data/"	
runInJMP = True # change to False if not running in JMP

if runInJMP:
	import jmputils
	jmputils.jpip('install --upgrade', 'pip setuptools')
	jmputils.jpip('install', 'pandas numpy scikit-learn keras lightgbm')
	jmputils.jpip('install', 'scikit-learn')
	jmputils.jpip('install', 'gpytorch')
	jmputils.jpip('install', 'prettytable')
	jmputils.jpip('install', 'tab_transformer_pytorch')
	#jmputils.jpip('install', 'pickle')
	
	
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# from tab9f, dim 8, FinalTable, X2024 markers, stdize within env, change loss 50*corr
mname = 'tab10a'
#path = './'

from pathlib import Path

# Get the absolute path of the current script as a Path object
script_path = Path(__name__).resolve()
# Get the directory containing the script
script_directory = script_path.parent
# Get the parent directory (one level up from the script's directory)
parent_directory = script_directory.parent
# Get the grand-parent directory (one level up from the script's directory)
grandparent_directory = parent_directory.parent
print(f"Script path: {script_path}")
print(f"Script directory: {script_directory}")
print(f"Parent directory: {parent_directory}")
print(f"Grandparent directory: {grandparent_directory}")

import os
#os.chdir(r'???')
import sys
lib_path = root_path + '\Code\M3'
sys.path.append(lib_path)

data_dir = os.path.join(grandparent_directory, 'Results/DataPrep/')
print(data_dir)

path_dir = os.path.join(grandparent_directory, 'Results/M3/')
print(path_dir)

# %%
IC = 'ic3'
HIDDEN = 500

KM = 'km1'
# NCOMP = 1000
NCOMP = 5000

MODE = 'cosine'
# MODE = 'euclidean'

# MARKER_PREFIX = 'q'
# MARKER_PREFIX = 'S'

# NUM_LEAVES = 51
# NCOMP = 20000
# STEP = 0.5

MAX_EPOCHS = 40
MIN_EPOCHS = 5
CONVERGENCE = 1e-3
PATIENCE = 10
# LR = 0.001
LR = 0.01
SEED = 40

# %%
# %env CUDA_DEVICE_ORDER=PCI_BUS_ID
# # %env CUDA_VISIBLE_DEVICES=0
# %env CUDA_VISIBLE_DEVICES=1

# %%
# # get current notebook name sans javascript
# # https://stackoverflow.com/questions/12544056/how-do-i-get-the-current-ipython-jupyter-notebook-name/52187331#52187331
# # use notebook name as model name when saving 

# from notebook import notebookapp
# import urllib
# import json
# import os
# import ipykernel

# def notebook_path(full_path=False):
#     """Returns the absolute path of the Notebook or None if it cannot be determined
#     NOTE: works only when the security is token-based or there is also no password
#     """
#     connection_file = os.path.basename(ipykernel.get_connection_file())
#     kernel_id = connection_file.split('-', 1)[1].split('.')[0]

#     for srv in notebookapp.list_running_servers():
#         try:
#             if srv['token']=='' and not srv['password']:  # No token and no password, ahem...
#                 req = urllib.request.urlopen(srv['url']+'api/sessions')
#             else:
#                 req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
#             sessions = json.load(req)
#             for sess in sessions:
#                 if sess['kernel']['id'] == kernel_id:
#                     if full_path: return os.path.join(srv['notebook_dir'],sess['notebook']['path'])
#                     else: return sess['notebook']['path'].split('/')[-1][:-6]
#         except:
#             pass  # There may be stale entries in the runtime directory 
#     return None

# mname = notebook_path()
mname

# %%
import pandas as pd
import numpy as np
import gc, os
import time
import random
pd.set_option('display.max_rows', 10)

# %%
# # !pip install gpytorch

# %%
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %%
# # !pip install pytorch_tabular[all]
# # !pip install torchmetrics==0.5
# # !pip install omegaconf==2.1.2
# # !pip install pytorch_tabnet
# # !pip install colorama
# # !pip install prettytable
# # !pip install rdflib
# # !pip install pyarrow
# # !pip install -U pandas
# # !pip install -U tqdm
# # !pip install ipywidgets
# # !pip install -U pytorch_lightning
# # !pip install lightgbm

# %%
# from pytorch_tabular import TabularModel
# from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, NodeConfig
# from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig

# %%
# cross-validation mlp ensemble 
# https://machinelearningmastery.com/how-to-create-a-random-split-cross-validation-and-bagging-ensemble-for-deep-learning-in-keras/
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.utils.extmath import randomized_svd
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# torch.multiprocessing.set_start_method('spawn')

# Tabnet 
# from pytorch_tabnet.metrics import Metric
# from pytorch_tabnet.tab_model import TabNetRegressor
# from abc import abstractmethod
# from pytorch_tabnet import tab_network
# from pytorch_tabnet.multiclass_utils import unique_labels
# from pytorch_tabnet.pretraining import TabNetPretrainer

# %%
### Make prettier output ###
from colorama import Fore
c_ = Fore.CYAN
m_ = Fore.MAGENTA
r_ = Fore.RED
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN

from prettytable import PrettyTable

import matplotlib.pyplot as plt
# from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax
import pickle

# %%
readCSV = True
if readCSV:
	a = pd.read_csv(data_dir+'FinalTable.csv', low_memory=False)
else:
	a = pd.read_pickle(data_dir+'FinalTable.pickle')	
print(a)
	
saveFinalPKL = False
if saveFinalPKL:
	fname = data_dir+'FinalTable.pickle'
	with open(fname, "wb") as f:
		pickle.dump(a, f)	

# %%
# standardize response within each environment
groups = a[['Env','Estimate']].groupby("Env")
mean_env, std_env = groups.transform("mean"), groups.transform("std")

a['Estimate'] = (a.Estimate - mean_env.Estimate) / std_env.Estimate
print(a)

# %%
markers1 = [c for c in a.columns if c.startswith('S1_')]
markers2 = [c for c in a.columns if c.startswith('S2_')]
markers3 = [c for c in a.columns if c.startswith('S3_')]
markers4 = [c for c in a.columns if c.startswith('S4_')]
markers5 = [c for c in a.columns if c.startswith('S5_')]
markers6 = [c for c in a.columns if c.startswith('S6_')]
markers7 = [c for c in a.columns if c.startswith('S7_')]
markers8 = [c for c in a.columns if c.startswith('S8_')]
markers9 = [c for c in a.columns if c.startswith('S9_')]
markers10 = [c for c in a.columns if c.startswith('S10_')]

markers11 = [c for c in a.columns if c.startswith('ADD')]
markers12 = [c for c in a.columns if c.startswith('DOM')]

markers = markers1 + markers2 + markers3 + markers4 + markers5 + markers6 + markers7 + markers8 + \
          markers9 + markers10 + markers11 + markers12
print(len(markers))

# %%
a = a.drop(columns=markers)
print(a)

# %%
x1 = pd.read_csv(data_dir+'X2024.csv')
print(x1)

# %%
a = pd.merge(a,x1, left_on='Hybrid', right_on='Unnamed: 0', how='left')
print(a)

# %%
markers = [c for c in a.columns if c.startswith('V')]
print(len(markers))

# %%
# set up categorical variables
cat_cols = ['Treatment1','Previous_Crop1','State','Station1']
cats = []
categories = []
for c in cat_cols:
    a[c] = a[c].astype('category')
    cnew = c+'_cat'
    a[cnew] = a[c].cat.codes
    cats.append(cnew)
    cn = a[cnew].nunique()
    categories.append(cn)
    print(cnew,cn)

# %%
print(cats, categories)

# %%
# continuous features
ll = pd.read_csv(data_dir+'G2F_Lat_Lon.csv')
latlon = ['Lat_WS','Lon_WS']
del ll['N Rows']
print(ll)

# %%
a = a.set_index('Experiment_Recode')

# %%
ll = ll.set_index('Experiment_Recode')

# %%
a = a.join(ll)

# %%
a = a.reset_index()
print(a)

# %%
# impute missing with mean
# b = b.fillna(b.mean())
# d = b[b.Year <= 2017]
# t = b[b.Year > 2017]
# print(d.shape, t.shape)
# d
a[markers] = a[markers].fillna(a[markers].mean())

# %%
features = markers + latlon
print(len(features))

# %%
t = a[a.Year == 2024].reset_index()
print(t)

# %%
# targets = list(a.columns[3:8])
targets = ['Estimate']
targets

# %%
a0 = a

# %%
a = a[a[targets[0]].notnull()].reset_index()
print(a.shape)

# %%
# a[lags] = a[lags].fillna(a[lags].mean())

# %%
a[targets].describe()

# %%
a[targets].plot.hist()

# %%
a[targets].isnull().mean()

# %%
# # label encode location
# a['location'] = a['location'].astype('category')
# a['location_cat'] = a['location'].cat.codes

# a['location_cat'].describe()

# %%
# cats = ['location_cat']
# categories = [41]
# cats, categories

# %%
# x = pd.get_dummies(a['location']).values
# x.shape

# %%
# impute missing with mean, could try more sophisticated methods
# a = a.fillna(a.mean())

# %%
# targets = list(a.columns[1:7])
# targets = list(a.columns[1:2])
# targets = list(a.columns[2:3])
# targets = list(a.columns[3:4])
# targets = list(a.columns[4:5])
# targets = list(a.columns[5:6])
# targets = list(a.columns[6:7])
# target = targets[0]
print(targets)

# %%
atargets = a[targets].reset_index(drop=True)

# %%
atargets.describe()

# %%
target = targets[0]
a[target].plot.hist()

# %%
a[target].isnull().mean()

# %%
ids = ['Env','Hybrid','Year']
print(ids)

# %%
# hold out each year
N_SPLITS = 10
N_REPEATS = 1
folds = []
for yr in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
     folds.append(
         (a[a.Year != yr].index.values,
          a[a.Year == yr].index.values))
folds[0], len(folds)

# %%
# N_SPLITS = 5
# N_REPEATS = 1
# folds = []
# for i in range(N_REPEATS):
#     # folds.extend(list(StratifiedKFold(n_splits=N_SPLITS, random_state=i, shuffle=True).split(X, y)))
#     folds.extend(list(KFold(n_splits=N_SPLITS, random_state=i, shuffle=True).split(a, y)))
# print(len(folds))

# %%
# x = b[features]
# y = a[target]
# print(y.shape)

# %%
# np.mean(x)

# %%
# np.mean(y)

# %%
os.makedirs(path_dir+'mod', exist_ok=True)
os.makedirs(path_dir+'oof', exist_ok=True)
os.makedirs(path_dir+'imp', exist_ok=True)
os.makedirs(path_dir+'sub', exist_ok=True)
os.makedirs(path_dir+'cvr', exist_ok=True)

# %%
import gc
# del a
gc.collect()

# %%
# # !pip install lightgbm --upgrade
# # !pip install xgboost --upgrade

# %%
# impute missing with mean
# a = a.fillna(a.mean())

# %%
# # # training and test data
# X = a[targets+markers].values
# # y = d[targets].values
# # X_test = t[features].values
# # print(X.shape, y.shape, X_test.shape)

# # np.mean(X, axis=0)

# # center and scale features, way faster than tabular
# m = np.mean(X,axis=0)
# s = np.std(X,axis=0) + 1e-8
# a = pd.DataFrame((X - m)/s, columns=targets+markers)

# %%
# ncomp = NCOMP
# U, S, Vt = randomized_svd(a[markers].values, n_components=ncomp, random_state=SEED)
# print(U.shape, S.shape, Vt.shape)

# # np.diag(S)
# q = pd.DataFrame(np.matmul(U, np.diag(S/np.sqrt(nm))),
#                  columns=['q'+str(i) for i in range(U.shape[1])], index=a.index)
# # q.shape
# # q

# a = pd.concat([atargets,a,q], axis=1)
# a

# %%
# a = pd.concat([atargets,a], axis=1)
print(a)

# %%
# fname = mname + '.pq'
# a.to_parquet(fname)
# print(fname, a.shape)

# %%
# a['q0'].plot.hist()

# %%
# nlist = []
# nlist = list(q.columns)

# %%
# features = nlist
# features = latlon + markers
# features = markers + nlist
# features = ['lat','lon'] + flist + nlist + laglist
# # features = ['lat','lon'] + markers + flist + nlist
nd = len(features)
print(features[:10], nd)

# %%
# a[target] = y.values
print(a.shape)
gc.collect()

# a[features] = (X - m)/s
# X = (X - m)/s
# X_test = (X_test - m)/s

# (s==0).mean()

# np.mean(X, axis=0)

# goodcols = ~np.all(np.isnan(X), axis=0)
# X = X[:,goodcols]
# X_test = X_test[:,goodcols]
# X.shape, X_test.shape

# features = [f for (i,f) in enumerate(features) if goodcols[i]]
# len(features)

# %%
# a[target].isnull().mean()

# %%
# num_col_names = ['lat','lon'] + nlist
num_col_names = features
# num_col_names = nlist + [m for m in markers if m.startswith(MARKER_PREFIX)]
cat_col_names = cats
# cat_col_names = ['fl','Male Pedigree','Female Pedigree']
print(len(num_col_names), num_col_names[:10], cat_col_names)

# %%
num_col_names[-10:]

# %%
a[num_col_names[-10:]].plot.hist()

# %%
print(folds[0])

# %%
a.iloc[folds[0][0]].Year.value_counts()

# %%
a.iloc[folds[0][1]].Year.value_counts()

# %%
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

# %%
# show parameters in pytorch model, requires prettytable
def count_parameters(model, print_table=True):
    if print_table:
        table = PrettyTable(["Module", "Parameters"])
        table.align['Module'] = 'l'
        table.align['Parameters'] = 'r'
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if print_table: table.add_row([name, param])
        total_params += param
    if print_table: print(table)
    print(f"Total Trainable Parameters: {total_params:,}\n")
#     return total_params

# %%
Dataset

# %%
# basic torch dataset
class BasicDataset(Dataset):
    def __init__(self, x, y, z, device='cuda'):
        # super(BasicDataset).__init__()
        self.x = x
        self.y = y
        self.z = z
        self.device = device
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        data = {
            # return torch tensors
            # 'x':torch.from_numpy(self.x[i]).float().to(device),
            # 'y':torch.from_numpy(self.y[i]).float().to(device),
            # 'z':torch.from_numpy(self.z[i]).float().to(device)
            
            'x':torch.from_numpy(self.x[i]).float(),
            'y':torch.from_numpy(self.y[i]).float(),
            'z':torch.from_numpy(self.z[i]).float()
            
            # return numpy arrays
            # 'x':self.x[i],
            # 'y':self.y[i],
            # 'z':self.z[i]
        }
        return data


# %%
#device = 'cuda'
device = 'cpu'

EPOCHS = MAX_EPOCHS

# NUM_BATCHES = 2
NUM_BATCHES = 5

# BATCH_SIZE = 1024
# BATCH_SIZE = 1024 * 4
# BATCH_SIZE = 1024 * 16
# BATCH_SIZE = 1024 * 64
BATCH_SIZE = len(a) // NUM_BATCHES + 1

BATCH_SIZE_TRAIN = BATCH_SIZE
BATCH_SIZE_VALID = BATCH_SIZE
BATCH_SIZE_TEST = BATCH_SIZE

# NUM_WORKERS_TRAIN = 14
# NUM_WORKERS_VALID = 14
# NUM_WORKERS_TEST = 14
NUM_WORKERS_TRAIN = 0
NUM_WORKERS_VALID = 0
NUM_WORKERS_TEST = 0

train_sampler = None
valid_sampler = None
test_sampler = None

train_shuffle = True
valid_shuffle = False
test_shuffle = False

torch.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)

n = a.shape[0]

# y = torch.from_numpy(a[[target]].values).float().to(device)
# x = torch.ones((n,1)).float().to(device)
# z = torch.from_numpy(a[features].values).float().to(device)

y = a[targets].values
# y = (y - y.mean())/ y.std()
# a[targets] = y
# x = np.ones((n,1))
x = a[cats].values
z = a[features].values

# %%
x_test = t[cats].values
y_test = np.repeat(np.nan,x_test.shape[0]).reshape(-1,1)
z_test = t[features].values

print(x_test.shape, y_test.shape, z_test.shape)

# %%
print(y_test)

# %%
# 0.01 is the default, raise to avoid warning message while training
# gpytorch.settings.cg_tolerance(0.02)

# %%
# # !pip install tab-transformer-pytorch

# %%
# import torch
# import torch.nn as nn
from tab_transformer_pytorch import TabTransformer


# %%
# cont_mean_std = torch.randn(10, 2)

# model = TabTransformer(
#     categories = (10, 5, 6, 5, 8),      # tuple containing the number of unique values within each category
#     num_continuous = 10,                # number of continuous values
#     dim = 32,                           # dimension, paper set at 32
#     dim_out = 1,                        # binary prediction, but could be anything
#     depth = 6,                          # depth, paper recommended 6
#     heads = 8,                          # heads, paper recommends 8
#     attn_dropout = 0.1,                 # post-attention dropout
#     ff_dropout = 0.1,                   # feed forward dropout
#     mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
#     mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
#     continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
# )

# x_categ = torch.randint(0, 5, (1, 5))     # category values, from 0 - max number of categories, in the order as passed into the constructor above
# x_cont = torch.randn(1, 10)               # assume continuous values are already normalized individually

# pred = model(x_categ, x_cont)

# %%
# lossfn = nn.MSELoss()

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

lossfn = RMSELoss

# # create a nn class (just-for-fun choice :-) 
# class RMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
        
#     def forward(self,yhat,y):
#         return torch.sqrt(self.mse(yhat,y))

# lossfn = RMSELoss()

def corrloss(target, pred):
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return 1.0 - (pred_n * target_n).sum()

corrlossfn = corrloss

print(tuple(categories))   

print(BATCH_SIZE_TRAIN)
print(NUM_WORKERS_TRAIN)
    
# %%
# k-fold cv loop
y_preds = []
t_preds = []

for fold_idx, fold in enumerate(folds):
    foldi = fold_idx // N_SPLITS
    foldj = fold_idx % N_SPLITS

    print()
    print('-'*25)
    print(f'{target}  rep {foldi}  fold {foldj}')
    print('-'*25)

    # fold = folds[FOLD]
    x_train, y_train, z_train = x[fold[0]], y[fold[0]], z[fold[0]]
    x_val, y_val, z_val = x[fold[1]], y[fold[1]], z[fold[1]]

    n_train, n_val = len(fold[0]), len(fold[1])
    nz = z.shape[1]

    print('x', x.shape, ' x_train', x_train.shape, ' x_val', x_val.shape)
    print('y', y.shape, ' y_train', y_train.shape, ' y_val', y_val.shape)
    print('z', z.shape, ' z_train', z_train.shape, ' z_val', z_val.shape)

    train_dataset = BasicDataset(x_train, y_train, z_train, device)
    valid_dataset = BasicDataset(x_val, y_val, z_val, device)
    test_dataset = BasicDataset(x_test, y_test, z_test, device)   

    trainloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler,
                                              num_workers = NUM_WORKERS_TRAIN,
                                              # pin_memory=True,
                                              batch_size=BATCH_SIZE_TRAIN, shuffle=train_shuffle)
    validloader = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler,
                                              num_workers = NUM_WORKERS_VALID,
                                              # pin_memory=True,
                                              batch_size=BATCH_SIZE_VALID, shuffle=valid_shuffle)
    testloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler,
                                          num_workers = NUM_WORKERS_TEST,
                                          # pin_memory=True,
                                          batch_size=BATCH_SIZE_TEST, shuffle=test_shuffle)
                                          
    print(f"Shape of z before forward pass: {z.shape[1]}")                            

    model = TabTransformer(
        categories = tuple(categories),     # tuple containing the number of unique values within each category
        num_continuous = z.shape[1],        # number of continuous values
        dim = 8,                            # dimension, paper set at 32
        dim_out = 1,                        # binary prediction, but could be anything
        depth = 6,                          # depth, paper recommended 6
        heads = 8,                          # heads, paper recommends 8
        attn_dropout = 0.1,                 # post-attention dropout
        ff_dropout = 0.1,                   # feed forward dropout
        mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        use_shared_categ_embed = False,  # FIX: Prevents adding 4 to your 32 dim
        # continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
    ).to(device)

    if fold_idx==0: count_parameters(model)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  

    loss_prev = torch.tensor(100000)
    loss_best = torch.tensor(1e12)
    not_best = 0                

    for epoch in range(EPOCHS):

        model.train()
        train_preds = []
        train_targs = []
        loss_train = 0
        n_batch = 0
        for data in trainloader:
            x_train, y_train, z_train = data['x'], data['y'].squeeze(), data['z']

            # select k best markers
            # sel = SelectKBest(f_regression, k=NCOMP)
            # sel = RFE(RandomForestRegressor(), n_features_to_select=NCOMP, verbose=1)
            # sel = RFE(lgb.LGBMRegressor(num_leaves=NUM_LEAVES, verbose=1),
            #           n_features_to_select=NCOMP,
            #           step=STEP, verbose=1)
            # z_train = sel.fit_transform(z_train.numpy(), y_train)
            # z_val = sel.transform(z_val.numpy())

            x_train = x_train.long().to(device)
            y_train = y_train.float().to(device)
            z_train = z_train.float().to(device)
            # z_train = torch.from_numpy(z_train).float().to(device)
           
            print(f"Shape of x before forward pass: {x_train.shape}")
            print(f"Shape of z before forward pass: {z_train.shape}")

            # forward pass to compute model predictions
            yp_train = model(x_train, z_train)
            train_preds.append(yp_train.detach().cpu().numpy())
            train_targs.append(y_train.cpu().numpy())
 
            # zero out gradients
            optimizer.zero_grad()

            # calc loss and backprop gradients
            loss = lossfn(yp_train.squeeze(), y_train.squeeze()) + \
                   50*corrlossfn(yp_train.squeeze(), y_train.squeeze())
            loss_train += loss

            # backward pass to compute gradients
            loss.backward()

            # take an optimization step
            optimizer.step()

            n_batch += 1

        loss_train /= n_batch
        yp_train = np.concatenate(train_preds)
        y_train = np.concatenate(train_targs)
        train_corr = np.corrcoef(yp_train.squeeze(),y_train.squeeze())[0,1]

        with torch.no_grad():
            model.eval()
            valid_preds = []
            valid_targs = []
            loss_valid = 0
            n_batch = 0
            for data in validloader:
                x_val, y_val, z_val = data['x'], data['y'].squeeze(), data['z']

                x_val = x_val.long().to(device)
                y_val = y_val.float().to(device)
                z_val = z_val.float().to(device)
                # z_val = torch.from_numpy(z_val).float().to(device)

                yp_val = model(x_val, z_val)
                valid_preds.append(yp_val.detach().cpu().numpy())
                valid_targs.append(y_val.cpu().numpy())
                loss = lossfn(yp_val.squeeze(), y_val.squeeze()) + \
                       50*corrlossfn(yp_val.squeeze(), y_val.squeeze())
                loss_valid += loss
                n_batch += 1
            loss_valid /= n_batch
            yp_val = np.concatenate(valid_preds)
            y_val = np.concatenate(valid_targs)
            val_corr = np.corrcoef(yp_val.squeeze(),y_val.squeeze())[0,1]

            
        # track and print progress
        # diff = loss.squeeze() - loss_prev
        if loss_valid < loss_best:
            star = '*'
            loss_best = loss_valid
            not_best = 0
            # save best model
            # fname = f'./mod/{mname}_{fold_idx}.pt'
            # torch.save(model.state_dict(), fname)
        else:
            star = ''
            not_best += 1

        if epoch % 1 == 0: print(
            epoch, 
            # f'{model.likelihood.noise.item():.4f}',
            # f'{model.mean_module.constant.item():.4f}',
            # f'{model.covar_module.raw_outputscale.item():.4f}',
            # f'{model.covar_module.base_kernel.raw_lengthscale.item():.4f}',
            f'{loss_train.item():.4f}',
            f'{loss_valid.item():.4f}{star}',
            f'{train_corr:.4f}',
            f'{val_corr:.4f}',
        )

        # loss_prev = loss
        # if ((np.abs(diff.data.cpu().numpy()) < CONVERGENCE) and (epoch >= MIN_EPOCHS)): break
        # if (not_best >= PATIENCE):
        #     print('early stopping')
        #     break    
            
    plt.scatter(x=yp_train.squeeze(), y=y_train.squeeze())
    plt.show()

    print(f'fold {fold_idx}  train corr {train_corr:.4f}',)
    print()

    plt.scatter(x=yp_val, y=y_val)
    plt.show()

    print(f'fold {fold_idx}  val corr {val_corr:.4f}')            

    y_preds.append( pd.DataFrame(yp_val, index = a.iloc[fold[1]].index, columns=[f'{mname}'] ) )

    # test set predictions
    with torch.no_grad():
        test_preds = []
        n_batch = 0
        for data in testloader:
            x_tst, y_tst, z_tst = data['x'], data['y'].squeeze(), data['z']
    
            x_tst = x_tst.long().to(device)
            y_tst = y_tst.float().to(device)
            z_tst = z_tst.float().to(device)
            # z_val = torch.from_numpy(z_val).float().to(device)
    
            yp_test = model(x_tst, z_tst)
            test_preds.append(yp_test.detach().cpu().numpy())
            n_batch += 1
        yp_tst = np.concatenate(test_preds)
        t_preds.append(yp_tst)
        # t_preds.append( pd.DataFrame(yp_tst, columns=[f'{mname}'] ) )
        print('test', yp_test.shape, len(t_preds))

# save oofs
yp = pd.concat(y_preds)
# yp = yp.set_index(a['index']).sort_index()
# yp = yp.groupby(yp.index).mean().squeeze()
fname = f'oof/{mname}.pkl'
yp.to_pickle(path_dir + fname)
print(fname, yp.shape)

# save test preds
print(t_preds)
tp = np.stack(t_preds)
print(tp[[[0]]])
fname = f'sub/{mname}.npy'
np.save(path_dir + fname, tp)
print(fname, tp.shape)

tp2 = tp.reshape(tp.shape[0], -1) #
print(tp2.shape)
print(tp2)

tp2t = tp2.T
print(tp2t.shape)
print(tp2t)

tp2d = pd.DataFrame(tp2t)
print(tp2d)

tp2dj = tp2d.join(t[['Env','Hybrid','Experiment_Recode','Year']])
print(tp2dj.shape)
print(tp2dj)

tp2dj.rename(columns={0: '2014', 1: '2015', 2: '2016', 3: '2017', 4: '2018', 5: '2019', 6: '2020', 7: '2021', 8: '2022', 9: '2023'}, inplace=True)
print(tp2dj)

fname = f'sub/{mname}.csv'
tp2dj.to_csv(path_dir + fname)
print(fname, tp2dj.shape)

tp2dj_stacked = tp2dj.set_index(['Env','Hybrid','Experiment_Recode','Year']).stack().reset_index()
print(tp2dj_stacked)
tp2dj_stacked.rename(columns={0: 'YPred', 'level_4': 'Fold'}, inplace=True)
tp2dj_stacked['EnvFold'] = tp2dj_stacked['Experiment_Recode'].str.cat(tp2dj_stacked['Fold'], sep="_")
tp2dj_stacked['EnvTest'] = tp2dj_stacked.apply(lambda row: f"{row['Experiment_Recode']}_{row['Year']}", axis=1)
print(tp2dj_stacked)

tp2dj_stacked.sort_values(by=['Fold', 'Hybrid'], inplace=True)
print(tp2dj_stacked)
fname = f'sub/{mname}_stacked.csv'
tp2dj_stacked.to_csv(path_dir + fname)
print(fname, tp2dj_stacked.shape)

# tp = pd.concat(t_preds)
# fname = f'sub/{mname}.pkl'
# tp.to_pickle(fname)
# print(fname, tp.shape)

print()
print()
yp = yp.groupby(yp.index).mean()
# yp.columns = [f'{mname}_{target}']
yp = yp.join(a[[target]])

yp.plot.scatter(x=f'{mname}', y=target)
plt.show()

print('oof corr', yp.corr())

# %%
yp.plot.scatter(x=f'{mname}', y=target)
plt.show()

print('oof corr', yp.corr())

# %%
print(yp)

# %%
print(yp.corr())

# %%
ypd = yp.join(a[['Env','Hybrid','Experiment_Recode','Year']])
print(ypd)

# %%
ypd = ypd.groupby(['Env','Hybrid','Experiment_Recode','Year']).mean().reset_index()
print(ypd)

# %%
ypd.plot.scatter(x=f'{mname}', y=target)
plt.show()

print('oof corr', ypd[[target,mname]].corr())

# %%
fname = 'oof/' + mname + '.pkl'
ypd[[mname]].to_pickle(path_dir + fname)
print(fname, ypd.shape)

# %%
fname = 'b_' + mname + '.pkl'
ypd[['Env','Hybrid']].to_pickle(path_dir + fname)
print(fname, ypd.shape)

# %%
fname = 'oof/' + mname + '_pred.csv'
ypd.to_csv(path_dir + fname)
print(fname, ypd.shape)

# %%
# Define a function to calculate grouped RMSE
def rmse(group):
    return np.sqrt(np.mean((group['Estimate'] - group[mname]) ** 2))

# %%
# Calculate RMSE by group
gby = ['Experiment_Recode','Year']
rmse_by_group = pd.DataFrame(ypd.groupby(gby).apply(rmse)).reset_index()

rmse_by_group.columns = gby + ['RMSE']

print(rmse_by_group)

# %%
# Calculate RMSE by group
rmse_by_group2 = pd.DataFrame(rmse_by_group[['Year','RMSE']].groupby(['Year']).mean()).reset_index()

# rmse_by_group2.columns = ['Year','AvgRMSE']

print(rmse_by_group2)

# %%
rmse_by_group2.plot.line(x='Year', y='RMSE')

# %%
# Define a function to calculate RMSE
def corr(group):
    return np.corrcoef(group['Estimate'], group[mname])[0,1]

# %%
# Calculate correlation by group
corr_by_group = pd.DataFrame(ypd.groupby(gby).apply(corr)).reset_index()

corr_by_group.columns = gby + ['Corr']

print(corr_by_group)

# %%
corr_by_group.Corr.describe()

# %%
corr_by_group2 = pd.DataFrame(corr_by_group[['Year','Corr']].groupby(['Year']).mean('Corr')).reset_index()
print(corr_by_group2)

print(corr_by_group2['Corr'].mean())

# %%
corr_by_group2.plot.line(x='Year', y='Corr')

# %%
import seaborn as sns

# %%
sns.boxplot(x='Year',y='Corr',data=corr_by_group)

# %%
plt.figure(figsize=(20, 12))
sns.boxplot(x='Experiment_Recode',y='Corr',data=corr_by_group)

# %%
