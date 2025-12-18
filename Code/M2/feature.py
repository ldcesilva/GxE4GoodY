'''
all steps from raw files to features, before any training
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
	#jmputils.jpip('install', 'PackageName') #add more packages if missing


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
#print(os.getcwd())

import sys
lib_path = root_path + '/Code/M2/'
sys.path.append(lib_path)
print(os.getcwd())

import json
import copy
from matplotlib import pyplot as plt
import time
import numpy as np
import pandas as pd
import random
from scipy.stats import pearsonr
import seaborn as sns
import pickle

from helpful import ven2, cp, rmse, evaluatepred
from helpful import getenv, checkclean, getdata, save_env2fea, path
from helpful import uniq
print(os.getcwd())


#%% all file paths
data_dir = os.path.join(grandparent_directory, 'Data')
print(data_dir)
#data_dir = os.path.abspath(data_dir)
#print(data_dir)

train_dir22   = 'Challenge2022/Training_data/'
train_dir24   = 'Challenge2024/Training_data/'
train_trait   = os.path.join(data_dir, train_dir22+'1_Training_Trait_Data_2014_2021.csv')
train_trait24 = os.path.join(data_dir,train_dir24+'1_Training_Trait_Data_2014_2023.csv')
#train_meta    = os.path.join(data_dir,train_dir22+'2_Training_Meta_Data_2014_2021.csv')
#train_meta2   = os.path.join(data_dir,train_dir22+'2_Training_Meta_Data_2014_2021_edit.csv')
#train_meta3   = os.path.join(data_dir,train_dir22+'2_Training_Meta_Data_2014_2021_fill.csv')
train_meta24  = os.path.join(data_dir,train_dir24+'2_Training_Meta_Data_2014_2023_fill.csv') 
train_meta24  = os.path.join(data_dir,train_dir24+'2_Training_Meta_Data_2014_2023.csv') #sbrlsi
train_soil    = os.path.join(data_dir,train_dir22+'3_Training_Soil_Data_2015_2021.csv')
train_soil24  = os.path.join(data_dir,train_dir24+'3_Training_Soil_Data_2015_2023.csv')
train_weat    = os.path.join(data_dir,train_dir22+'4_Training_Weather_Data_2014_2021.csv')
train_weat24  = os.path.join(data_dir,train_dir24+'4_Training_Weather_Data_2014_2023_seasons_only.csv')
train_ec      = os.path.join(data_dir,train_dir22+'6_Training_EC_Data_2014_2021.csv')
train_ec24    = os.path.join(data_dir,train_dir24+'6_Training_EC_Data_2014_2023.csv')

f_geno      = os.path.join(data_dir,train_dir22+'5_Genotype_Data_All_Years.vcf')
f_geno24    = os.path.join(data_dir,train_dir24+'5_Genotype_Data_All_2014_2025_Hybrids.vcf')
f_geno24n   = os.path.join(data_dir,train_dir24+'5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt')
f_hybrid    = os.path.join(data_dir,train_dir22+'All_hybrid_names_info.csv')
f_source    = os.path.join(data_dir,train_dir22+'GenoDataSources.txt')
f_sourcebio = os.path.join(data_dir,train_dir22+'GenoDataSourcesWithUpdatedBioProject.txt')

test_dir22    = 'Challenge2022/Testing_data/'
test_dir24    = 'Challenge2024/Testing_data/'
test_sub    = os.path.join(data_dir,test_dir22+'1_Submission_Template_2022.csv')
test_sub24  = os.path.join(data_dir,test_dir24+'1_Submission_Template_2024.csv')
#test_meta   = os.path.join(data_dir,test_dir22+'2_Testing_Meta_Data_2022.csv')
#test_meta2  = os.path.join(data_dir,test_dir22+'2_Testing_Meta_Data_2022_edit.csv')
#test_meta3  = os.path.join(data_dir,test_dir22+'2_Testing_Meta_Data_2022_fill.csv')
test_meta24 = os.path.join(data_dir,test_dir24+'2_Testing_Meta_Data_2024_fill.csv')
test_meta24 = os.path.join(data_dir,test_dir24+'2_Testing_Meta_Data_2024.csv')
test_soil   = os.path.join(data_dir,test_dir22+'3_Testing_Soil_Data_2022.csv')
test_soil24 = os.path.join(data_dir,test_dir24+'3_Testing_Soil_Data_2024.csv')
test_weat   = os.path.join(data_dir,test_dir22+'4_Testing_Weather_Data_2022.csv')
test_weat24 = os.path.join(data_dir,test_dir24+'4_Testing_Weather_Data_2024_seasons_only.csv')
test_ec     = os.path.join(data_dir,test_dir22+'6_Testing_EC_Data_2022.csv')
test_ec24   = os.path.join(data_dir,test_dir24+'6_Testing_EC_Data_2024.csv')
test_gt     = os.path.join(data_dir,test_dir22+'Test_Set_Observed_Values_ANSWER.csv')

#data_path = 'data_2024/' # store intermediate data 
data_path = os.path.join(grandparent_directory, 'Results/M2') # store intermediate data 
print(data_path)

#%% read all
df_main  = pd.read_csv(train_trait)
df_tar   = pd.read_csv(test_sub)
df_main2 = pd.read_csv(train_trait24)
df_tar2  = pd.read_csv(test_sub24)

dftr_meta = pd.read_csv(train_meta24)
dfte_meta = pd.read_csv(test_meta24)
dftr_soil = pd.read_csv(train_soil24)
dfte_soil = pd.read_csv(test_soil24)
dftr_ec_  = pd.read_csv(train_ec)
dftr_ec   = pd.read_csv(train_ec24)
dfte_ec   = pd.read_csv(test_ec24)
dftr_weat = pd.read_csv(train_weat24) 
dfte_weat = pd.read_csv(test_weat24) 

# should consider train~test together 
df_meta = pd.concat([dftr_meta, dfte_meta], ignore_index=True)
df_soil = pd.concat([dftr_soil, dfte_soil], ignore_index=True)
df_ec   = pd.concat([dftr_ec, dfte_ec], ignore_index=True) 
df_weat = pd.concat([dftr_weat, dfte_weat], ignore_index=True) 


#%% sanity check first

train_env = getenv(df_main)
print('raw feature - Env overlap')
ven2(train_env, getenv(df_meta))
ven2(train_env, getenv(df_soil))
ven2(train_env, getenv(df_weat))
ven2(train_env, getenv(df_ec))

train_env2 = getenv(df_main2)
print('raw feature - Env overlap')
ven2(train_env2, getenv(df_meta))
ven2(train_env2, getenv(df_soil))
ven2(train_env2, getenv(df_weat))
ven2(train_env2, getenv(df_ec))

#%% final training data shouldn't be fully complete, but should useful

fea_main = ['Env','Year','Hybrid','Yield_Mg_ha']
"""
fea_meta = ['Treatment','City','Farm',
            'Weather_Station_Latitude (in decimal numbers NOT DMS)',
            'Weather_Station_Longitude (in decimal numbers NOT DMS)',
            'Previous_Crop','Pounds_Needed_Soil_Moisture',
            'Cardinal_Heading_Pass_1']
"""
fea_meta = ['Treatment','City','Farm',
            'Weather_Station_Latitude (in decimal numbers NOT DMS)',
            'Weather_Station_Longitude (in decimal numbers NOT DMS)',
            'Previous_Crop','Pounds_Needed_Soil_Moisture'] #remove Cardinal_Heading_pass_1 because it is a mix of float and string           
fea_soil = ['E Depth','1:1 Soil pH',
            'WDRF Buffer pH','1:1 S Salts mmho/cm','Texture No',
            'Organic Matter LOI %','Nitrate-N ppm N','lbs N/A',
            'Potassium ppm K','Sulfate-S ppm S','Calcium ppm Ca',
            'Magnesium ppm Mg','Sodium ppm Na','CEC/Sum of Cations me/100g',
            '%H Sat','%K Sat','%Ca Sat','%Mg Sat','%Na Sat',
            'Mehlich P-III ppm P','% Sand','% Silt','% Clay','Texture']
fea_weat = df_weat.columns[2:].tolist() # 'RH2M' to end

fea_ec   = df_ec.columns[1:].tolist() # 'HI30_pGerEme' to end
ec_not   = ['yield_pGerEme', 'yield_pEmeEnJ', 'yield_pEnJFlo', 'yield_pFloFla']
fea_ec   = [i for i in fea_ec if i not in ec_not ]
'''
ec_not = ['Env']
com_ec = np.intersect1d(df_ec.columns, dftr_ec_.columns)
fea_ec   = [i for i in com_ec if i not in ec_not ]
'''

fea_all = fea_main + fea_meta + fea_soil + fea_weat + fea_ec
print('all features:', len(fea_all))
# all features: 706

# sanity check
assert len(fea_all) == len(set(fea_all))

#%% sanity check first
checkclean(df_meta, fea_meta) # check all
checkclean(df_soil, fea_soil) # check all
# pay attention to those show once/twice

checkclean(df_weat, fea_weat)
checkclean(df_ec,     fea_ec)

#%% save both train/test features
data_meta = getdata(df_meta, fea_meta) 
data_soil = getdata(df_soil, fea_soil) 

### save features
save_env2fea(data_meta, path(data_path, 'data_meta.pkl'))
save_env2fea(data_soil, path(data_path, 'data_soil.pkl'))

#%% ========== ========== ========== ========== ========== ==========

# weather feature

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import tsfel

# only seasonal, get from 2024: contains train/test
print('df_weat:', df_weat.shape)

train_envs = uniq(df_main['Env'])
tar_envs   = uniq(df_tar['Env'])
ven2(train_envs, uniq(df_weat['Env']))
ven2(tar_envs, uniq(df_weat['Env']))

train_envs2 = uniq(df_main2['Env'])
tar_envs2   = uniq(df_tar2['Env'])
ven2(train_envs2, uniq(df_weat['Env']))
ven2(tar_envs2, uniq(df_weat['Env']))

#%% see weather trend
env   = train_envs2[-1]
_wea_ = df_weat[df_weat['Env']==env]
print('env:{}, data:{}'.format(env,len(_wea_)))

fig, ax = plt.subplots()
ax.plot(_wea_.iloc[:,2:])

#%% append timestamp
df_weat['time'] = -1
weat_envs = uniq(df_weat['Env'])

for env in weat_envs:
    _wea_ = df_weat[df_weat['Env']==env].iloc[:,2:] # no: Env, Date
    _len_ = len(_wea_)
    print('env:{}, data:{}'.format(env,_len_))
    df_weat.loc[df_weat['Env']==env,'time'] = np.arange(_len_)

assert df_weat['time'].values.min() == 0
#%% 2024 weather imputation
for col in df_weat.columns:
    df_weat[col] = df_weat[col].fillna(method='ffill')

#%% extract all weather features
extracted_features = extract_features(df_weat.drop('Date', axis=1),
                                      column_id = "Env", 
                                      column_sort = "time",
                                      n_jobs = 1)
# must be: feature cols + ID + time
print(extracted_features.shape)
# (269, 12528)

impute(extracted_features)

#%% extract tsfel features
weat_dim = 16
cfg = tsfel.get_features_by_domain()
fs = 7 # per week

tsfel_features = []
for env in weat_envs:
    _wea_ = df_weat[df_weat['Env']==env].iloc[:,2:-1] # no: Env, Date, time
    assert _wea_.shape[1] == weat_dim
    _len_ = len(_wea_)
    print('env:{}, data:{}'.format(env,_len_))
    _f_ = tsfel.time_series_features_extractor(cfg, _wea_, fs=fs)
    tsfel_features.append(_f_.to_numpy()[0])

#%%
tsfel_features = np.array(tsfel_features)
print('tsfel_features:', tsfel_features.shape)

combined_features = np.hstack((extracted_features.values, tsfel_features))
print('combined_features:', combined_features.shape)

#%% find unique values number
v_uniq = []
for i in range(combined_features.shape[1]):
    unique_v, counts = np.unique(combined_features[:,i], return_counts=True)
    v_uniq.append(len(unique_v))
v_uniq = np.array(v_uniq)

# delete features with single value
combined_features = combined_features[:, v_uniq > 1]
print('combined_features:', combined_features.shape)

#%% save complete weather features
data_weat_raw = dict(zip(weat_envs, combined_features))
save_env2fea(data_weat_raw, path(data_path, 'data_weat_raw.pkl'))

#%% ========== ========== ========== ========== ========== ==========

# ec feature
dataec = df_ec.iloc[:,1:] # loc[:,fea_ec] # only keep common features
envsec = df_ec.iloc[:,0]

#%% save complete ec
data_ec_raw = dict(zip(envsec, dataec.values))
save_env2fea(data_ec_raw, path(data_path, 'data_ec_raw.pkl'))

#%% ========== ========== ========== ========== ========== ==========

# complete features - imputation

def load_env2fea(fname):
    with open(fname, "rb") as f:
        data_meta = pickle.load(f)
    return data_meta 

getdim = lambda data_meta : len(next(iter(data_meta.values())))

data_meta     = load_env2fea(path(data_path, 'data_meta.pkl'))
data_soil     = load_env2fea(path(data_path, 'data_soil.pkl'))
data_weat_raw = load_env2fea(path(data_path, 'data_weat_raw.pkl'))
data_ec_raw   = load_env2fea(path(data_path, 'data_ec_raw.pkl'))

dim_meta = getdim(data_meta)
dim_soil = getdim(data_soil)
dim_weat = getdim(data_weat_raw)
dim_ec   = getdim(data_ec_raw)

print('dim_meta:',dim_meta)
print('dim_soil:',dim_soil)
print('dim_weat:',dim_weat)
print('dim_ec:',  dim_ec)

''' 
dim_meta: 8
dim_soil: 24
dim_weat: 14196
dim_ec: 581
'''
#%% see overlap before imputation
df_main = pd.read_csv(train_trait)
train_envs = uniq(df_main['Env'])

# NOTE: when doing imputation, should do for both train/test
df_tar    = pd.read_csv(test_sub)
test_envs = uniq(df_tar['Env'])

all_envs = train_envs + test_envs

ven2(all_envs, list(data_meta.keys()))
ven2(all_envs, list(data_soil.keys()))
ven2(all_envs, list(data_weat_raw.keys()))
ven2(all_envs, list(data_ec_raw.keys()))

# 2024 data
df_main2 = pd.read_csv(train_trait24)
train_envs2 = uniq(df_main2['Env'])

# NOTE: when doing imputation, should do for both train/test
df_tar2    = pd.read_csv(test_sub24)
test_envs2 = uniq(df_tar2['Env'])

all_envs2 = train_envs2 + test_envs2

ven2(all_envs2, list(data_meta.keys()))
ven2(all_envs2, list(data_soil.keys()))
ven2(all_envs2, list(data_weat_raw.keys()))
ven2(all_envs2, list(data_ec_raw.keys()))

#%% imputation all
from helpful import data_impu, assimpu

data_soil_impu = data_soil     | data_impu(all_envs, data_soil)
data_weat_impu = data_weat_raw | data_impu(all_envs, data_weat_raw)
data_ec_impu   = data_ec_raw   | data_impu(all_envs, data_ec_raw)

assimpu(data_soil,     data_soil_impu)
assimpu(data_weat_raw, data_weat_impu)
assimpu(data_ec_raw,   data_ec_impu)

ven2(all_envs2, list(data_meta.keys()))
ven2(all_envs2, list(data_soil_impu.keys()))
ven2(all_envs2, list(data_weat_impu.keys()))
ven2(all_envs2, list(data_ec_impu.keys()))

#%% save imputation
save_env2fea(data_soil_impu, path(data_path, 'data_soil_impu.pkl'))
save_env2fea(data_weat_impu, path(data_path, 'data_weat_impu.pkl'))
save_env2fea(data_ec_impu,   path(data_path, 'data_ec_impu.pkl'))

# imputation DONE

#%%
#%% ========== ========== ========== ========== ========== ==========
