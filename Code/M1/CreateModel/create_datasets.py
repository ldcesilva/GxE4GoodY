# Modified from https://github.com/igorkf/Maize_GxE_Prediction/tree/main/src/create_datasets.py
root_path = "//tdl/Public2/G2F/Manuscript/" # Adjust to your path
prep_path = root_path + "/Results/DataPrep/"
train_path = root_path + "/Data/Challenge2024/Training_data/"
test_path = root_path + "/Data/Challenge2024/Testing_data/"	
runInJMP = True # change to False if not running in JMP

if runInJMP:
	import jmp
	import jmputils

	jmputils.jpip('install --upgrade', 'pip setuptools')
	jmputils.jpip('install', 'pandas numpy scikit-learn keras lightgbm')
	jmputils.jpip('install', 'scikit-learn')
	
import sys
import argparse
from pathlib import Path
import random

import pandas as pd
from sklearn.decomposition import TruncatedSVD

# adding to the system path
sys.path.append(root_path + '/Code/M1/CreateModel')

from preprocessing import (
    process_metadata,
    process_test_data,
    lat_lon_to_bin,
#    create_folds,
    agg_yield,
    process_blues,
    feat_eng_weather,
    feat_eng_soil,
    feat_eng_target,
    extract_target,
    create_field_location
) 

OUTPUT_PATH = Path(root_path + "/Results/M1/")
print(OUTPUT_PATH)
TRAIT_DATA_PATH = train_path + '1_Training_Trait_Data_2014_2023.csv'
TEST_DATA_PATH = test_path + '/1_Submission_Template_2024.csv'
META_TRAIN_PATH = train_path + '/2_Training_Meta_Data_2014_2023.csv'
META_TEST_PATH = test_path + '/2_Testing_Meta_Data_2024.csv'

META_COLS = ['Env', 'weather_station_lat', 'weather_station_lon', 'treatment_not_standard']
CAT_COLS = ['Env', 'Hybrid']  # to avoid NA imputation

LAT_BIN_STEP = 1.2
LON_BIN_STEP = LAT_BIN_STEP * 3

#if __name__ == '__main__':
 
# META
meta = process_metadata(META_TRAIN_PATH)
print(meta.head())
meta_test = process_metadata(META_TEST_PATH)

# TEST
test = process_test_data(TEST_DATA_PATH)
xtest = test.merge(meta_test[META_COLS], on='Env', how='left').drop(['Field_Location'], axis=1)
df_sub = xtest.reset_index()[['Env', 'Hybrid']]

# TRAIT
trait = pd.read_csv(TRAIT_DATA_PATH)
trait = trait.merge(meta[META_COLS], on='Env', how='left')
trait = create_field_location(trait)
print(trait.head())

# agg yield (unadjusted means)
trait = agg_yield(trait)
print(trait.head())

# WEATHER
weather = pd.read_csv(prep_path + '/Subset Train 56 days 1_Training_Trait with 1_Training_LSMeans with Concat_Meta with Concat_Weather.csv')
weather_test = pd.read_csv(prep_path + '/Subset Test 56 days 1_Training_Trait with 1_Training_LSMeans with Concat_Meta with Concat_Weather.csv')

# SOIL
soil = pd.read_csv(prep_path + '/3_Training_Soil_Data_2015_2023.csv')
soil_test = pd.read_csv(prep_path + '/3_Testing_Soil_Data_2024.csv')

# EC
ec = pd.read_csv(prep_path + '/6_Training_EC_Data_2014_2023.csv').set_index('Env')
ec_test = pd.read_csv(prep_path + '/6_Testing_EC_Data_2024.csv').set_index('Env')   

xtrain = trait.copy()
del xtrain['Field_Location']
del xtrain['Year']

# replace unadjusted means by BLUEs
blues = pd.read_csv(prep_path + '/_1_training_lsmeans.csv')
blues = blues.rename(columns={'Estimate': 'predicted.value'})
del blues['Effect'], blues['StdErr'], blues['DF'], blues['tValue'], blues['Probt']
print(blues.head())
print(blues.shape)

xtrain = xtrain.merge(blues, on=['Env', 'Hybrid'], how='left')
print(xtrain.shape)
xtrain = process_blues(xtrain)	
print(xtrain.shape)

# feat eng (weather)
weather_feats = feat_eng_weather(weather)
weather_test_feats = feat_eng_weather(weather_test)
xtrain = xtrain.merge(weather_feats, on='Env', how='left')
xtest = xtest.merge(weather_test_feats, on='Env', how='left')    

# feat eng (soil)
xtrain = xtrain.merge(feat_eng_soil(soil), on='Env', how='left')
xtest = xtest.merge(feat_eng_soil(soil_test), on='Env', how='left')

# feat eng (EC)
xtrain_ec = ec[ec.index.isin(xtrain['Env'])].copy()
xtest_ec = ec_test[ec_test.index.isin(xtest['Env'])].copy()

n_components = 15
seed = 123
svd = TruncatedSVD(n_components=n_components, n_iter=20, random_state=seed)
svd.fit(xtrain_ec)
print('SVD explained variance:', svd.explained_variance_ratio_.sum())

xtrain_ec = pd.DataFrame(svd.transform(xtrain_ec), index=xtrain_ec.index)
component_cols = [f'EC_svd_comp{i}' for i in range(xtrain_ec.shape[1])]
xtrain_ec.columns = component_cols
xtest_ec = pd.DataFrame(svd.transform(xtest_ec), columns=component_cols, index=xtest_ec.index)    

xtrain = xtrain.merge(xtrain_ec, on='Env', how='left')
xtest = xtest.merge(xtest_ec, on='Env', how='left')
 
for dfs in [xtrain, xtest]:
    dfs['T2M_std_spring_X_weather_station_lat'] = dfs['T2M_std_spring'] * dfs['weather_station_lat']
    dfs['T2M_std_spring_X_weather_station_lat'] = dfs['T2M_std_summer'] * dfs['weather_station_lat']

    # binning lat/lon seems to help reducing noise
    dfs['weather_station_lat'] = dfs['weather_station_lat'].apply(lambda x: lat_lon_to_bin(x, LAT_BIN_STEP))
    dfs['weather_station_lon'] = dfs['weather_station_lon'].apply(lambda x: lat_lon_to_bin(x, LON_BIN_STEP))
    
print('lat/lon unique bins:')
print('lat:', sorted(set(xtrain['weather_station_lat'].unique())))
print('lon:', sorted(set(xtrain['weather_station_lon'].unique())))

# remove NA phenotype if needed
xtrain = xtrain[~xtrain['Yield_Mg_ha'].isnull()].reset_index(drop=True)

# set index
xtrain = xtrain.set_index(['Env', 'Hybrid'])
xtest = xtest.set_index(['Env', 'Hybrid'])

# extract targets
ytrain = extract_target(xtrain)
ytest = extract_target(xtest)

print(xtrain.isnull().sum() / len(xtrain))
print(xtest.isnull().sum() / len(xtest))

print(CAT_COLS)
print(xtrain.columns)

# NA imputing
for col in [x for x in xtrain.columns if x not in CAT_COLS]:
    mean = xtrain[col].mean()
    xtrain[col].fillna(mean, inplace=True)
    xtest[col].fillna(mean, inplace=True)

print('xtrain shape:', xtrain.shape)
print('xtest shape:', xtest.shape)
print('ytrain shape:', ytrain.shape)
print('ytrain nulls:', ytrain.isnull().sum() / len(ytrain))

assert xtrain.index.names == ['Env', 'Hybrid']

# write datasets
xtrain.reset_index().to_csv(OUTPUT_PATH / f'gxe_xtrain.csv', index=False)
ytrain.reset_index().to_csv(OUTPUT_PATH / f'gxe_ytrain.csv', index=False)
xtest.reset_index().to_csv(OUTPUT_PATH / f'gxe_xtest.csv', index=False)
ytest.reset_index().to_csv(OUTPUT_PATH / f'gxe_ytest.csv', index=False)
