'''
run first - feature.py
run second - mean-yield prediction
    - save ids, save predicted mean-yield
this - prepare genotype + env feature table
'''
root_path = "//tdl/Public2/G2F/Manuscript/" # Adjust to your path
prep_path = root_path + "/Results/DataPrep/"
train_path = root_path + "/Data/Challenge2024/Training_data/"
test_path = root_path + "/Data/Challenge2024/Testing_data/"	
runInJMP = True # change to False if not running in JMP

if runInJMP:
	import jmputils
	jmputils.jpip('install --upgrade', 'pip setuptools')
	jmputils.jpip('install', 'matplotlib')
	jmputils.jpip('install', 'seaborn')
	jmputils.jpip('install', 'statsmodels')
	jmputils.jpip('install', 'tqdm')
	jmputils.jpip('install', 'catboost')
	jmputils.jpip('install', 'torch')
	jmputils.jpip('install', 'tsfresh')
	jmputils.jpip('install', 'tsfel')
	
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
#os.chdir(root_path)
import sys
lib_path = root_path + '\Code\M2'
sys.path.append(lib_path)

import copy
from matplotlib import pyplot as plt
import time
import numpy as np
import pandas as pd
import random
from scipy.stats import pearsonr
import seaborn as sns
import pickle
import json
from tqdm import tqdm

from helpful import ven2, cp, rmse, evaluatepred
from helpful import path, uniq

#%%

data_dir = os.path.join(grandparent_directory, 'Data')
print(data_dir)

train_dir22     = 'Challenge2022/Training_data/'
train_dir24     = 'Challenge2024/Training_data/'
train_trait22   = os.path.join(data_dir, train_dir22+'1_Training_Trait_Data_2014_2021.csv')
train_trait24   = os.path.join(data_dir, train_dir24+'1_Training_Trait_Data_2014_2023.csv')
f_geno24n       = os.path.join(data_dir, train_dir24+'5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt')

test_dir22   = 'Challenge2022/Testing_data/'
test_dir24   = 'Challenge2024/Testing_data/'
test_sub22   = os.path.join(data_dir, test_dir22+'1_Submission_Template_2022.csv')
test_sub24   = os.path.join(data_dir, test_dir24+'1_Submission_Template_2024.csv')
test_gt22    = os.path.join(data_dir, test_dir22+'Test_Set_Observed_Values_ANSWER.csv')

#data_path = 'data_2024/' # store intermediate data 
data_path = os.path.join(grandparent_directory, 'Results/M2') # store intermediate data 
data_path_orig = data_path

#%% ========== ========== ========== ========== ========== ==========

# first prepare genotype-env datatable

# load complete feature sets
def load_env2fea(fname):
    with open(fname, "rb") as f:
        data_meta = pickle.load(f)
    return data_meta 

getdim = lambda data_meta : len(next(iter(data_meta.values())))

data_meta = load_env2fea(path(data_path, 'data_meta.pkl'))
data_soil = load_env2fea(path(data_path, 'data_soil_impu.pkl'))
data_weat = load_env2fea(path(data_path, 'data_weat_impu.pkl'))
data_ec   = load_env2fea(path(data_path, 'data_ec_impu.pkl'))

dim_meta = getdim(data_meta)
dim_soil = getdim(data_soil)
dim_weat = getdim(data_weat)
dim_ec   = getdim(data_ec)

print('dim_meta:',dim_meta)
print('dim_soil:',dim_soil)
print('dim_weat:',dim_weat)
print('dim_ec:',  dim_ec)

#%% fit mean-yield
version = 2024 # 2022 or 2024

if version == 2022:
    print('using data:', version)
    df_main = pd.read_csv(train_trait22)
    train_envs = uniq(df_main['Env'])
elif version == 2024:
    print('using data:', version)
    df_main = pd.read_csv(train_trait24)
    train_envs = uniq(df_main['Env'])

#%%
ven2(train_envs, list(data_meta.keys()))
ven2(train_envs, list(data_soil.keys()))
ven2(train_envs, list(data_weat.keys()))
ven2(train_envs, list(data_ec.keys()))

for key, value in data_meta.items():
    print(f"{key}: {value}")
    
#%%
name_meta = ['m_'+str(i) for i in range(dim_meta)]
name_soil = ['s_'+str(i) for i in range(dim_soil)]
name_weat = ['w_'+str(i) for i in range(dim_weat)]
name_ec   = ['e_'+str(i) for i in range(dim_ec)]    

#%% load selected feature indices
"""
fname = path(data_path, str(version)+'/env_selected_ids.pkl')
with open(fname, "rb") as f:
    old_summary = pickle.load(f)
    old_id      = pickle.load(f)

snames = old_summary['selected_features_names']

#%%
if 'dy' not in snames:
    snames.append('dy')
   
"""
snames = name_meta + name_soil + name_weat + name_ec
print('pre-selected features:', len(snames))
print(snames)

#%%
def check_np(data_meta):
    for k,v in data_meta.items():
        assert type(v) == np.ndarray

check_np(data_meta)
check_np(data_soil)
check_np(data_weat)
check_np(data_ec)

#%%
sid_meta = [idx for idx,i in enumerate(name_meta) if i in snames]
sid_soil = [idx for idx,i in enumerate(name_soil) if i in snames]
sid_weat = [idx for idx,i in enumerate(name_weat) if i in snames]
sid_ec   = [idx for idx,i in enumerate(name_ec  ) if i in snames]

feat_meta = {k:v[sid_meta] for k,v in data_meta.items()}
feat_soil = {k:v[sid_soil] for k,v in data_soil.items()}
feat_weat = {k:v[sid_weat] for k,v in data_weat.items()}
feat_ec   = {k:v[sid_ec]     for k,v in data_ec.items()}

dim_meta = getdim(feat_meta)
dim_soil = getdim(feat_soil)
dim_weat = getdim(feat_weat)
dim_ec   = getdim(feat_ec)

ven2(train_envs, list(feat_meta.keys()))
ven2(train_envs, list(feat_soil.keys()))
ven2(train_envs, list(feat_weat.keys()))
ven2(train_envs, list(feat_ec.keys()))

#%%
geno24 = pd.read_csv(f_geno24n, skiprows=1, sep='\t')
geno24 = geno24.set_index('<Marker>').T

dim_geno = len(geno24)

#%%
env_hyb2dat = {}
env_hyb2yie = {}
for index, row in df_main.iterrows():
    e = row['Env']
    h = row['Hybrid']
    y = row['Year']
    e_h = e+h
    yie = row['Yield_Mg_ha']
    if e_h not in env_hyb2yie:
        env_hyb2yie[e_h] = []
        env_hyb2dat[e_h] = [e, y, h]
    if not np.isnan(yie):
        env_hyb2yie[e_h].append(yie)

#%% mean yield
datl = []
for env_hyb, ys in env_hyb2yie.items():
    if len(ys) > 0:
        _info_ = env_hyb2dat[env_hyb]
        datl.append(_info_ + [np.mean(ys)])

#%% using full items [optional]
df_ = df_main.dropna(subset=['Yield_Mg_ha'], inplace=False)
print('removing nan: {} -> {}'.format(len(df_main), len(df_)))

_e = df_['Env'].values
_h = df_['Hybrid'].values
_y = df_['Year'].values
_t = df_['Yield_Mg_ha'].values

datl = [[e,y,h,t] for e,h,y,t in zip(_e,_h,_y,_t)]
print(datl[0])

num_evar = dim_meta + dim_soil + dim_weat + dim_ec
print(num_evar)
print(dim_geno)
print(num_evar + dim_geno)

print(feat_meta.keys())

#%%
feal = []
ally = []
for index,item in enumerate(datl):
    e    = item[0]
    year = item[1]
    h    = item[2]
    yie  = item[3]    
    
    _fea_ = [e, h] # record env, hybrid
    _mis_ = ''
    
    # feature - meta
    if e in feat_meta.keys():
        _fea_meta = feat_meta[e]
    else:
        _fea_meta = np.full(dim_meta, np.nan)
        _mis_ += '!meta '
    _fea_ += _fea_meta.tolist()
      
    # feature - soil
    if e in feat_soil.keys():
        _fea_soil = feat_soil[e]
    else:
        _fea_soil = np.full(dim_soil, np.nan)
        _mis_ += '!soil '
    _fea_ += _fea_soil.tolist()
    
    # feature - weat
    if e in feat_weat.keys():
        _fea_weat = feat_weat[e]
    else:
        _fea_weat = np.full(dim_weat, np.nan)
        _mis_ += '!weat '
    _fea_ += _fea_weat.tolist()
    
    # feature - ec
    if e in feat_ec.keys():
        _fea_ec = feat_ec[e]
    else:
        _fea_ec = np.full(dim_ec, np.nan)
        _mis_ += '!ec '
    _fea_ += _fea_ec.tolist()
    
    # feature - added
    _fea_ += [e[:4], e[:3], year-2013.]
    
    # feature - geno
    if h in geno24.columns:
        _fea_geno = geno24[h].values
    else:
        # _fea_geno = np.full(dim_geno, np.nan)
        # _mis_ += '!geno'
        continue
    _fea_ += _fea_geno.tolist()
    
    assert len(_fea_) == dim_meta + dim_soil + dim_weat + dim_ec + dim_geno + 3 + 2
        
    feal.append(_fea_)
    ally.append([e, h, yie])
    
    if len(_mis_) > 0:
        print(index, e, h, _mis_)
        
    #print(index, e, h, _mis_)   
    #if index == 5:
	#    break       

print((ally[0]))
        
print(len(snames))    
print(dim_geno)   
print(len(snames) + dim_geno + 2) 
print(len(feal[0]))
f0 = feal[0]
print(f0[14878:14888])
print(datl[0])

#%%
#datallfea = pd.DataFrame(feal, columns=['Env', 'Hybrid']+snames+['g_'+str(i) for i in range(dim_geno)])
datallfea = pd.DataFrame(feal, columns=['Env', 'Hybrid']+snames+['Env4bit', 'Env3bit', 'dy']+['g_'+str(i) for i in range(dim_geno)])
del feal
print(datallfea.shape)
datallfea = datallfea.drop_duplicates(subset=['Env', 'Hybrid'])
print(datallfea.shape)

#%% _ for full items
'''
datallfea -
    Env Hybrid
    m_ s_ w_ e_
    Env4bit Env3bit dy
    g_
    [83457 rows x 3425 columns]
'''
ffinal = path(data_path, str(version)+'/data_final_.pkl')
datallfea.to_pickle(ffinal)

Y = pd.DataFrame(ally, columns=['Env', 'Hybrid', 'mean_yield'])
print(Y.shape)
Y = Y.drop_duplicates(subset=['Env', 'Hybrid'])
print(Y.shape)
assert len(datallfea) == len(Y)
Y = Y['mean_yield']
ffinal = path(data_path, str(version)+'/label_final_.pkl')
Y.to_pickle(ffinal)

del datallfea

#%% ========== ========== ========== ========== ========== ==========

# then prepare submission data

if version == 2022:
    print('using data:', version)
    df_tar = pd.read_csv(test_sub22)
    al_env = df_tar['Env'].unique()
    al_hyb = df_tar['Hybrid'].unique()
elif version == 2024:
    print('using data:', version)
    df_tar = pd.read_csv(test_sub24)
    al_env = df_tar['Env'].unique()
    al_hyb = df_tar['Hybrid'].unique()

#%%
ven2(al_env, list(feat_meta.keys()))
ven2(al_env, list(feat_soil.keys()))
ven2(al_env, list(feat_weat.keys()))
ven2(al_env, list(feat_ec.keys()))
ven2(al_hyb, list(geno24.columns))

#%%
feal = []
for index, row in df_tar.iterrows():
    e    = row['Env']
    h    = row['Hybrid']    
    year = 2022. if version == 2022 else 2024.
    
    _fea_ = [e, h] # record env, hybrid
    _mis_ = ''
    
    # feature - meta
    if e in feat_meta.keys():
        _fea_meta = feat_meta[e]
    else:
        _fea_meta = np.full(dim_meta, np.nan)
        _mis_ += '!meta '
    _fea_ += _fea_meta.tolist()
    
    # feature - soil
    if e in feat_soil.keys():
        _fea_soil = feat_soil[e]
    else:
        _fea_soil = np.full(dim_soil, np.nan)
        _mis_ += '!soil '
    _fea_ += _fea_soil.tolist()
    
    # feature - weat
    if e in feat_weat.keys():
        _fea_weat = feat_weat[e]
    else:
        _fea_weat = np.full(dim_weat, np.nan)
        _mis_ += '!weat '
    _fea_ += _fea_weat.tolist()
    
    # feature - ec
    if e in feat_ec.keys():
        _fea_ec = feat_ec[e]
    else:
        _fea_ec = np.full(dim_ec, np.nan)
        _mis_ += '!ec '
    _fea_ += _fea_ec.tolist()
    
    # feature - added
    _fea_ += [e[:4], e[:3], year-2013.]
    
    # feature - geno
    if h in geno24.columns:
        _fea_geno = geno24[h].values
    else:
        # _fea_geno = np.full(dim_geno, np.nan)
        # _mis_ += '!geno'
        continue
    _fea_ += _fea_geno.tolist()
    
    assert len(_fea_) == dim_meta + dim_soil + dim_weat + dim_ec + dim_geno + 3 + 2
        
    feal.append(_fea_)
    
    if len(_mis_) > 0:
        print(index, e, h, _mis_)

#%%
#datallfea = pd.DataFrame(feal, columns=['Env', 'Hybrid']+snames+['g_'+str(i) for i in range(dim_geno)])
datallfea = pd.DataFrame(feal, columns=['Env', 'Hybrid']+snames+['Env4bit', 'Env3bit', 'dy']+['g_'+str(i) for i in range(dim_geno)])
print(datallfea.shape)
datallfea = datallfea.drop_duplicates(subset=['Env', 'Hybrid'])
print(datallfea.shape)

#%% save
'''
datallfea -
    Env Hybrid
    m_ s_ w_ e_
    Env4bit Env3bit dy
    g_
    [11555 rows x 3425 columns]
'''
ffinal = path(data_path, str(version)+'/data_final_submission.pkl')
datallfea.to_pickle(ffinal)

#%% ========== ========== ========== ========== ========== ==========

# fit yield 

from catboost import CatBoostRegressor, Pool, metrics, cv
from sklearn.model_selection import train_test_split
from catboost import EShapCalcType, EFeaturesSelectionAlgorithm
from helpful import select_features_syntetic

# load prepared data
if version == 2022:
    print('using data:', version)
    data_path = data_path_orig+'/2022/'
elif version == 2024:
    print('using data:', version)
    data_path = data_path_orig+'/2024/'
print(data_path)   
    
ffinal = path(data_path, 'data_final_.pkl') # _ for full items
X      = pd.read_pickle(ffinal)  
print('loaded X:', X.shape)
#X.rename(columns={'Field': 'Env4bit', 'FieldState': 'Env3bit'}, inplace=True)

ffinal = path(data_path, 'label_final_.pkl') # _ for full items
Y      = pd.read_pickle(ffinal)  
print('loaded Y:', Y.shape)

ffinal = path(data_path, 'data_final_submission.pkl')
Xt     = pd.read_pickle(ffinal)  
print('loaded Xt:', Xt.shape)
#Xt.rename(columns={'Field': 'Env4bit', 'FieldState': 'Env3bit'}, inplace=True)

suX  = X.count().sum()
suXt = Xt.count().sum()

if version == 2022:
    df_tar     = pd.read_csv(test_sub22)
    gt         = pd.read_csv(test_gt22)
    envhyb_sub = (df_tar['Env']+df_tar['Hybrid']).values
    envhyb_gt  = (gt['Env']+gt['Hybrid']).values.tolist()
    ven2(envhyb_sub, envhyb_gt)
    idmap      = [envhyb_gt.index(envhyb) if envhyb in envhyb_gt else 0 for envhyb in envhyb_sub]
    ygt        = gt['Yield_Mg_ha'].values[idmap]
    # plt.plot(ygt)
    Yt         = pd.Series(ygt, name='Yield_Mg_ha')
    # load main 
    df_main = pd.read_csv(train_trait22)
    train_envs = list(set(df_main['Env'].values.tolist()))
elif version == 2024:
    df_tar     = pd.read_csv(test_sub24)
    '''
    with open(path(data_path, 'guess_gtest.pkl'), "rb") as f:
        ygt = pickle.load(f)
    Yt = pd.Series(ygt, name='Yield_Mg_ha')
    assert len(Yt) == len(Xt)
    '''
    # load main
    df_main = pd.read_csv(train_trait24)
    train_envs = list(set(df_main['Env'].values.tolist()))

#%% use relative Yr
env2mean = {}

for env in train_envs:
    _d_ = df_main[df_main['Env']==env]    
    _v_ = _d_['Yield_Mg_ha'].dropna().values
    _y_ = np.mean(_v_)    
    print('{} in train-set:{}'.format(env, len(_d_)))
    env2mean[env] = _y_

print(env2mean.keys()) 
print(len(env2mean.keys()))     

X_envs = X['Env'].values
assert np.mean([(env in env2mean) for env in X_envs]) == 1
print(len(X_envs))

for idx,env in enumerate(X_envs):
	print(idx, env)
	if idx == 50:
		break
		
print(len(Y))		

#rela_yie = [Y[idx]-env2mean[env] for idx,env in enumerate(X_envs)] #error
rela_yie = Y
Yr       = pd.Series(rela_yie, name='rYield_Mg_ha')

''''
#%% append this mean yield
X['meanYield'] = [env2mean[env] for env in X_envs]
'''

'''
# read predicted mean yield
fname = path(data_path, 'pred_mean_yie.pkl')
with open(fname, "rb") as f:
    pred_mean_yie = pickle.load(f)

# NOTE: before use data_path to switch 22/24, however, here we have 9./11.
if version == 2022:
    print('using data:', version)
    assert set(Xt['dy']) == set([9.]) # i.e. only 2022 in our test data
    # append <predicted> mean yield
    Xt['meanYield'] = [pred_mean_yie[env+'_2022'] for env in Xt['Env4bit']]
elif version == 2024:
    print('using data:', version)
    assert set(Xt['dy']) == set([11.]) # i.e. only 2024 in our test data
    # append <predicted> mean yield
    Xt['meanYield'] = [pred_mean_yie[env+'_2024'] for env in Xt['Env4bit']]
'''
    
'''
#%% # first see most common values
for col in X.columns: 
    if col.startswith('g_'):
        continue
        print(X[col].mode()[0])
    elif X[col].dtype != float:
        continue
        print(X[col].mode()[0])
    else:
        print(X[col].mean(), '|', X[col].min(), X[col].max())
        if X[col].min() == X[col].max():
            print(X[col].min(), X[col].max())
'''            

#%% fill nan
print('before fill nan: {:.4f} {:.4f}'.format(X.isna().sum().sum()/suX, Xt.isna().sum().sum()/suXt))
for col in X.columns: # fill
    mis1 = X[col].isna().any()
    mis2 = Xt[col].isna().any()
    if mis1 or mis2:
        # find most common in X
        if col.startswith('g_'):
            _v_ = X[col].mode()[0] # most common genotype
        elif X[col].dtype != float:
            _v_ = X[col].mode()[0] # most common string
        else:
            _v_ = X[col].mean() # mean value
        if mis1:
            X[col].fillna(_v_, inplace=True)
            print(_v_, end='(1) ')
        if mis2:
            Xt[col].fillna(_v_, inplace=True)
            print(_v_, end='(2) ')

print('after fill nan: {:.4f} {:.4f}'.format(X.isna().sum().sum()/suX, Xt.isna().sum().sum()/suXt))

'''
#%% manual normalization [only for DL]
for col in X.columns:
    if col.startswith('g_'):
        continue # not for genos
    elif X[col].dtype != float:
        continue # not for categorical
    else:
        mi = X[col].min()
        ma = X[col].max()
        if mi != ma:
            X[col]  = (X[col].to_numpy() - mi) / (ma - mi)
            Xt[col] = (Xt[col].to_numpy() - mi) / (ma - mi)
            print(col, '|', mi, ma)
        
#%% also norm Yr [only for DL]
maY = abs(Yr).max() # 15.195867394884367
Yr  = Yr / maY
'''

# note: do NOT do norm for catboost

#%% save Yr for binning [full items]
np.save(path(data_path, 'y_train.npy'), np.array(rela_yie))

#%% drop Env Hybrid
XEnvHy = X[['Env', 'Hybrid']].copy()
X.drop(['Env', 'Hybrid'], axis=1, inplace=True)
Xt.drop(['Env', 'Hybrid'], axis=1, inplace=True)

#%%
print(X.dtypes)
categorical_features_indices = np.where(X.dtypes != float)[0]
print(X.columns[categorical_features_indices])

'''
#%% split: but use env to split  
rati    = 0.8
X_envs  = (X['Env4bit']+X['dy'].astype(str)).values
print(len(X_envs))
X_envsu = list(set(X_envs))

subset = random.sample(X_envsu, int(rati*len(X_envsu)))

testIDs  = [idx for idx,i in enumerate(X_envs) if i in subset]
trainIDs = [idx for idx,i in enumerate(X_envs) if i not in subset]
print('selected train:', len(trainIDs), ', test:', len(testIDs))

train_X = X.loc[trainIDs, :]
train_y = Yr.loc[trainIDs] # Y Yr
test_X  = X.loc[testIDs, :]
test_y  = Yr.loc[testIDs] # Y Yr

# rei = np.random.permutation(train_X.index)
# train_X = train_X.reindex(rei)
# train_y = train_y.reindex(rei)

#%%
train_pool = Pool(train_X, 
                  train_y, 
                  cat_features=categorical_features_indices,
                  has_header=True)
test_pool  = Pool(test_X, 
                  test_y,
                  cat_features=categorical_features_indices,
                  has_header=True)

#%%
synthetic_shap_summary = select_features_syntetic(train_pool,
                                                  test_pool,
    algorithm = EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
    loss = 'RMSE',
    raw_dim = X.shape[1],
    select = 2000
    )

ids = synthetic_shap_summary['selected_features']
'''

#%% 
params = {
    'iterations': 900,    
    'learning_rate': 0.03,
    'early_stopping_rounds': 150,
    'task_type': 'CPU',
    'use_best_model': False, # False, True
    # 'loss_function': 'Quantile:alpha=0.75',
}
model = CatBoostRegressor(**params)

#%% train using ALL features
model.fit(    
    X, Yr, # Yr
    cat_features = categorical_features_indices,    
    #eval_set = (Xt, Yt),
    logging_level = 'Verbose', 
    #plot = True
);   

'''
X_  = X
Xt_ = Xt

#%% train using selected features
X_  = X.iloc[:,ids]
Xt_ = Xt.iloc[:,ids]
categorical_features_indices_ = np.where(X_.dtypes != float)[0]

# rei = np.random.permutation(X.index)
# X_ = X.reindex(rei)
# Y_ = Yr.reindex(rei)

model.fit(
    X_, Yr, # Yr
    cat_features = categorical_features_indices_,    
    eval_set = (Xt_, Yt),
    logging_level = 'Verbose', 
    plot = True
);  

#%% use mean-yield + relative yield
predictionst = model.predict(Xt_) # * maY

mean_yie = [pred_mean_yie[env] for env in df_tar['Env'].values]
mean_yie = np.array(mean_yie)

predictionst = (predictionst + mean_yie)/1.

pred = df_tar.copy(deep=True)
pred['Yield_Mg_ha'] = predictionst

env2pr = evaluatepred(pred, gt)

#%% run 2024 model -> target guessed label
assert version == 2024
predictionst = model.predict(Xt_) # * maY

# Yt - guessed label - is already relative 
va_r = pearsonr(predictionst, Yt)[0]
plt.scatter(predictionst, Yt.values, s=15, alpha=0.6)
plt.xlabel('predicted')
plt.ylabel('guessed mean')
plt.title('pr: {:.4f}'.format(va_r))


#%%
cat_pred = path(data_path, 'cat_pred.npy')
np.save(cat_pred, predictionst)
'''

#%% predict on train data

predictions = model.predict(X)
print(predictions)
print(len(predictions))

with open(path(data_path, 'guess_gtrain.pkl'), "wb") as f:
    pickle.dump(predictions, f)
    
print(XEnvHy.head)    
print(XEnvHy.shape)    
pred = pd.DataFrame(predictions, columns=['ypred'])   
print(pred.head)
pred = pd.concat([XEnvHy.reset_index(drop=True), pred.reset_index(drop=True)], axis=1, ignore_index=True)
print(pred.shape)
pred.rename(columns={0: 'Env', 1 : 'Hybrid', 2: 'ypred'}, inplace=True)
print(pred.head)
pred.to_csv(os.path.join(data_path, 'guess_gtrain.csv'), index=False, header=True)

#%% predict on 2024 test data -> generate guess label

#data_path = 'data_2024/2024/'
data_path = os.path.join(data_path_orig, '2024') # store intermediate data 
ffinal = path(data_path, 'data_final_submission.pkl')
Xt     = pd.read_pickle(ffinal)  
print('loaded Xt:', Xt.shape)
#Xt.rename(columns={'Field': 'Env4bit', 'FieldState': 'Env3bit'}, inplace=True)
XtEnvHy = Xt[['Env', 'Hybrid']].copy()

'''
fname = path(data_path, 'pred_mean_yie.pkl')
with open(fname, "rb") as f:
    pred_mean_yie = pickle.load(f)

assert set(Xt['dy']) == set([11.]) # i.e. only 2022 in our test data
# append <predicted> mean yield
Xt['meanYield'] = [pred_mean_yie[env+'_2024'] for env in Xt['Env4bit']]
'''

for col in X.columns: # fill
    mis1 = False
    mis2 = Xt[col].isna().any()
    if mis1 or mis2:
        # find most common in X
        if col.startswith('g_'):
            _v_ = Xt[col].mode()[0] # most common genotype
        elif X[col].dtype != float:
            _v_ = Xt[col].mode()[0] # most common string
        else:
            _v_ = Xt[col].mean() # mean value        
        if mis2:
            Xt[col].fillna(_v_, inplace=True)
            print(_v_, end='(2) ')

Xt.drop(['Env', 'Hybrid'], axis=1, inplace=True)

predictionst = model.predict(Xt)
print(predictionst)
print(len(predictionst))

with open(path(data_path, 'guess_gtest.pkl'), "wb") as f:
    pickle.dump(predictionst, f)

predt = pd.DataFrame(predictionst, columns=['ypred'])   
predt = pd.concat([XtEnvHy, predt], axis=1)
print(predt.shape)
print(predt.head)
predt.to_csv(os.path.join(data_path, 'guess_gtest.csv'), index=False, header=True)
    
'''    
#%% ========== ========== ========== ========== ========== ==========

# export to DL

# using selected features
X_  = X.iloc[:,ids]
Xt_ = Xt.iloc[:,ids]
categorical_features_indices_ = np.where(X_.dtypes != float)[0]

#%% using all features for dl (but no categorical)
print(X.dtypes)
categorical_features_indices = np.where(X.dtypes != float)[0]
print(X.columns[categorical_features_indices])

X_  = X.drop(X.columns[categorical_features_indices], axis=1, inplace=False)
Xt_ = Xt.drop(Xt.columns[categorical_features_indices], axis=1, inplace=False)
print(X_.shape, Xt_.shape)

categorical_features_indices_ = np.where(X_.dtypes != float)[0]

#%%
#now we have:
#    X_ - selected X, Yr
#    Xt_ - selected Xt, Yt

rati    = 0.1
X_envs  = (X['Env4bit']+X['dy'].astype(str)).values
X_envsu = list(set(X_envs))

subset = random.sample(X_envsu, int(rati*len(X_envsu)))

testIDs  = [idx for idx,i in enumerate(X_envs) if i in subset]
trainIDs = [idx for idx,i in enumerate(X_envs) if i not in subset]
print('selected train:', len(trainIDs), ', test:', len(testIDs))

# train_X = X.loc[trainIDs, :]
# train_y = Yr.loc[trainIDs] # Y Yr 
val_X  = X_.loc[testIDs, :]
val_y  = Yr.loc[testIDs] # Y Yr 

assert len(categorical_features_indices_) == 0
assert X_.shape[1] == Xt_.shape[1] and X_.shape[1] == val_X.shape[1]
print('data split: {} train, {} val, {} test'.format(len(X_), len(val_X), len(Xt_)), '#features:', X_.shape[1])

#%% save to file
def save2dl(fn, x):
    print('saving:', x.shape)
    np.save(fn, x)

def savejs(fn, d):
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)
        
#%%
dirdl = data_path #'/home/chunma/dev/g2fdl/datasets/g2fdl_24_full/' # 'g2fdl/'
save2dl(dirdl+'N_train.npy', X_)
save2dl(dirdl+'N_test.npy',  X_) # should be N_val.npy
save2dl(dirdl+'N_val.npy',   Xt_) # should be N_test.npy

save2dl(dirdl+'y_train.npy', Yr) # Y Yr
save2dl(dirdl+'y_test.npy',  Y) # should be y_val.npy
save2dl(dirdl+'y_val.npy',   Yt) # should be y_test.npy

dlinfo = {"name": "g2f",
          "task_type": "regression",
          "n_classes": 1,
          "n_num_features": X_.shape[1],
          "n_cat_features": 0,
          "train_size": len(X_),
          "val_size": len(Xt_),
          "test_size": len(X_) }
savejs(dirdl+'info.json', dlinfo)
print(dlinfo)

#%% ========== ========== ========== ========== ========== ==========

# see DL result

predmlp = np.load('pred_4.npy')
assert len(predmlp) == len(Yt)

pred = df_tar.copy(deep=True)
pred['Yield_Mg_ha'] = predmlp + mean_yie

env2pr = evaluatpred(pred, gt)

#%% ========== ========== ========== ========== ========== ==========

# error analysis
df_train  = pd.read_csv(path(data_path, 'df_train.csv'))
df_test   = pd.read_csv(path(data_path, 'df_test.csv'))

# collect infomation
predictionst = model.predict(Xt_)
df_test['mean_yield'] = mean_yie
df_test['catboost'] = predictionst

predictionst = model.predict(X_)
df_train['mean_yield'] = X_['meanYield']
df_train['catboost'] = predictionst

df_train.to_csv('error_analysis/df_train_cat.csv',index=False, header=True)
df_test.to_csv('error_analysis/df_test_cat.csv',  index=False, header=True)

#%%
#%% ========== ========== ========== ========== ========== ==========
#%%
'''