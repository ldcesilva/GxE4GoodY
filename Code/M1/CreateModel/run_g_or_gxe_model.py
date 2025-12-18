# Modified from https://github.com/igorkf/Maize_GxE_Prediction/tree/main/src/run_g_or_gxe_model.py

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
	jmputils.jpip('install', 'pyarrow')

import numpy as np
import gc
import argparse
from pathlib import Path

import pandas as pd
import lightgbm as lgbm
from sklearn.decomposition import TruncatedSVD

# adding to the system path
sys.path.append(root_path + '/Code/M1/CreateModel')

from preprocessing import create_field_location
from evaluate import create_df_eval, avg_rmse, feat_imp

def preprocess_g(df, kinship, individuals: list):
    #print("preprocess_g")
    #df.columns = [x[:len(x) // 2] for x in df.columns]  # fix duplicated column names
    df.columns = [x for x in df.columns]  # fix duplicated column names
    df.index = df.columns
    df = df[df.index.isin(individuals)]  # filter rows
    df = df[[col for col in df.columns if col in individuals]]  # filter columns
    df.index.name = 'Hybrid'
    df.columns = [f'{x}_{kinship}' for x in df.columns]
    return df

def preprocess_kron(df, kinship):
    df[['Env', 'Hybrid']] = df['id'].str.split(':', expand=True)
    df = df.drop('id', axis=1).set_index(['Env', 'Hybrid'])
    df.columns = [f'{x}_{kinship}' for x in df.columns]
    # print(df.info(), '\n')
    # df[df.columns] = np.array(df.values, dtype=np.float32)  # downcast is too slow
    # print(df.info(), '\n')
    return df

def prepare_gxe(kinship):
    kron = pd.read_feather(OUTPUT_PATH / f'kronecker_{kinship}.arrow')
    kron = preprocess_kron(kron, kinship=kinship)
    return kron  
      
OUTPUT_PATH = Path(root_path + "/Results/M1/")
print(OUTPUT_PATH)

parser = argparse.ArgumentParser(prog="G2F")
parser.add_argument('--seed', type=int)
parser.add_argument('--A', action='store_true', default=False)
parser.add_argument('--D', action='store_true', default=False)
parser.add_argument('--E', action='store_true', default=False)
parser.add_argument('--AEI', action='store_true', default=False)
parser.add_argument('--DEI', action='store_true', default=False)
parser.add_argument('--svd', action='store_true', default=False)
parser.add_argument('--std', action='store_true', default=False)
parser.add_argument('--eplus', action='store_true', default=False)
parser.add_argument('--drop', action='store_true', default=False)
parser.add_argument('--cv', action='store_true', default=False)
parser.add_argument('--kf', action='store_true', default=False)
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--lag_features', action='store_true', default=False)
args = parser.parse_args()

args.A = True
args.D = True
args.E = True
args.AEI = False
args.DEI = False
args.svd = False
args.std = False
args.drop = False
args.eplus = False
args.cv = True
args.kf = False

hold22and23 = False
hold20and21 = False

if not args.A and not args.D and not args.E and not args.AEI and not args.DEI:
	raise Exception('Choose at least type of matrix: A, D, E, AEI, or DEI.')    
  
outfile = OUTPUT_PATH / f'oof_model'
print(outfile)   

"""
if __name__ == '__main__':
"""

# load targets
if args.std:
    outfile = f'{outfile}_std'   
    ytraindf = pd.read_csv(OUTPUT_PATH / f'gxe_ytrain_std.csv')
else:
    ytraindf = pd.read_csv(OUTPUT_PATH / f'gxe_ytrain.csv')
	
ytestdf = pd.read_csv(OUTPUT_PATH / f'gxe_ytest.csv')
ydf = ytraindf.copy()
ydf['Year'] = ydf['Env'].str[-4:].astype('int')	


if args.drop: 
    outfile = f'{outfile}_drop1819'   
    ydf = ydf.query('Year != 2018 and Year != 2019')

cv_labels = ydf['Year'].unique().tolist()
cv_labels.sort()
print(cv_labels)

#ydf['Year'] = ydf.index #cannot be put here
myCVIterator = []
if args.cv:
    if not args.kf:
        for i in cv_labels:
            if i != 2024:
                trainIndex = ydf[np.logical_and(ydf['Year'] != i, ydf['Year'] != 2024)].index.values.astype(int) 
                valIndex = ydf[np.logical_and(ydf['Year'] == i, ydf['Year'] != 2024)].index.values.astype(int)
                testIndex = ydf[ydf['Year'] == 2024].index.values.astype(int) 
                myCVIterator.append((trainIndex, valIndex)) 
        del ydf['Year'] 
    else:
        outfile = f'{outfile}_kf'
        if hold22and23:
            outfile = f'{outfile}_no22and23'
            y22and23 = ydf.query('Year == 2022 or Year == 2023')
            tyears = y22and23['Year'].unique().tolist()
            tyears.sort()
            print(tyears)
            del y22and23['Year']
            print(y22and23)
            ytestdf = [y22and23, ytestdf]
            ytestdf = pd.concat(ytestdf).reset_index(drop = True)
            ydf = ydf.query('Year != 2022 and Year != 2023')
            ydf = ydf.reset_index(drop = True)
            tryears = ydf['Year'].unique().tolist()
            tryears.sort()
            print(tryears)    
        elif hold20and21:
            outfile = f'{outfile}_no20and21'
            y20and21 = ydf.query('Year == 2020 or Year == 2021')
            tyears = y20and21['Year'].unique().tolist()
            tyears.sort()
            print(tyears)
            del y20and21['Year']
            print(y20and21)
            ytestdf = [y20and21, ytestdf]
            ytestdf = pd.concat(ytestdf).reset_index(drop = True)
            ydf = ydf.query('Year != 2020 and Year != 2021')
            ydf = ydf.reset_index(drop = True)
            tryears = ydf['Year'].unique().tolist()
            tryears.sort()
            print(tryears)                

        from sklearn.model_selection import KFold, StratifiedKFold
        N_FOLDS = 10      
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)   
        myCVIterator = folds.split(ydf['Yield_Mg_ha'])#enumerate(folds.split(ydf['Yield_Mg_ha']))   
        print(myCVIterator)  
        del ydf['Year']                    
else:
    '''
    print("Full")
    outfile = f'{outfile}_full'    
    trainIndex = ydf[ydf['Year'] != 2024].index.values.astype(int) 
    valIndex = ydf[ydf['Year'] != 2024].index.values.astype(int)
    testIndex = ydf[ydf['Year'] == 2024].index.values.astype(int) 
    myCVIterator.append((trainIndex, valIndex))  
    '''
    print("FullNo22and23")
    outfile = f'{outfile}_fullno22and23'    
    trainIndex = ydf[np.logical_and(ydf['Year'] != 2022, ydf['Year'] != 2023, ydf['Year'] != 2024)].index.values.astype(int) 
    valIndex = ydf[np.logical_or(ydf['Year'] == 2022, ydf['Year'] == 2023)].index.values.astype(int)
    testIndex = ydf[ydf['Year'] == 2024].index.values.astype(int) 
    myCVIterator.append((trainIndex, valIndex))  
    del ydf['Year'] 
    
#print(trainIndex)
#print(valIndex)
#print(testIndex)
print(myCVIterator)   
print(outfile)   
print(ydf)    
  
cv_count = 1 
concat_val_pred = []
concat_test_pred = []
for (trainIdx, valIdx) in myCVIterator:
    print("Fold: ", cv_count)
    print(trainIdx)
    print(valIdx)
    #ytrain, yval, ytest = ydf.loc[trainIdx], ydf.loc[valIdx], ydf.loc[testIdx] 
    ytrain, yval = ydf.iloc[trainIdx], ydf.iloc[valIdx]
	
    ytest = ytestdf.copy()

    # individuals
    individuals = ytrain['Hybrid'].unique().tolist() + yval['Hybrid'].unique().tolist() + ytest['Hybrid'].unique().tolist()
    individuals = list(dict.fromkeys(individuals))  # take unique but preserves order (python 3.7+)
    print('# unique individuals:', len(individuals))
    #print('# unique individuals:', individuals)

    if args.A or args.D or args.AEI or args.DEI:
        # load kinships or kroneckers
        kinships = []
        kroneckers = []
        if args.A or args.AEI:
            if args.A:
                print('Using A matrix.')
                if cv_count == 1:
                    outfile = f'{outfile}_A'
                A = pd.read_csv(OUTPUT_PATH / f'Additive.txt', sep=',')
                #print(A.shape)
                A = preprocess_g(A, 'A', individuals)
                #print(A.shape)
                kinships.append(A)
            if args.AEI:
                print('Using AEI matrix.')
                if cv_count == 1:
                    outfile = f'{outfile}_AEI'
                kroneckers.append(prepare_gxe('additive'))                
        #break          

        if args.D or args.DEI:
            if args.D:
                print('Using D matrix.')
                if cv_count == 1:
                    outfile = f'{outfile}_D'
                D = pd.read_csv(OUTPUT_PATH / f'Dominance.txt', sep=',')
                D = preprocess_g(D, 'D', individuals)
                kinships.append(D)
            if args.DEI:
                print('Using DEI matrix.')
                if cv_count == 1:
                    outfile = f'{outfile}_DEI'            
                kroneckers.append(prepare_gxe('dominance'))
                
    if (args.A and len(kinships) == 0) or (args.D and len(kinships) == 0) or (args.AEI and len(kroneckers) == 0) or (args.DEI and len(kroneckers) == 0):
        raise Exception('Choose at least one G or GE matrix.')                

    if args.E:       
        print('Using E matrix.')
        if not args.eplus:  
            if cv_count == 1:
                outfile = f'{outfile}_E'  
            Etraindf = pd.read_csv(OUTPUT_PATH / f'gxe_xtrain.csv')     
            #print(Etraindf)
            Etestdf = pd.read_csv(OUTPUT_PATH / f'gxe_xtest.csv')   
        else:
            if cv_count == 1:
                outfile = f'{outfile}_EPlus'
            Etraindf = pd.read_csv(OUTPUT_PATH / f'gxe_xtrain_plus.csv')     
            Etestdf = pd.read_csv(OUTPUT_PATH / f'gxe_xtest_plus.csv')                   
        #Edf = [Etraindf, Etestdf]           
        #Edf = pd.concat(Edf).reset_index(drop = True) 
        Edf = Etraindf.copy()
        if args.drop:
            Edf['Year'] = Edf['Env'].str[-4:].astype('int')
            Edf = Edf.query('Year != 2018 and Year != 2019')
            del Edf['Year']    
        #Etrain, Eval, Etest = Edf.loc[trainIdx], Edf.loc[valIdx], Edf.loc[testIdx] 
        
        if hold22and23:
            Edf['Year'] = Edf['Env'].str[-4:].astype('int')
            e22and23 = Edf.query('Year == 2022 or Year == 2023')
            del e22and23['Year']
            Etestdf = [e22and23, Etestdf]
            Etestdf = pd.concat(Etestdf).reset_index(drop = True)
            Edf = Edf.query('Year != 2022 and Year != 2023')
            Edf = Edf.reset_index(drop = True)  
            del Edf['Year'] 
        elif hold20and21:
            Edf['Year'] = Edf['Env'].str[-4:].astype('int')
            e20and21 = Edf.query('Year == 2020 or Year == 2021')
            del e20and21['Year']
            Etestdf = [e20and21, Etestdf]
            Etestdf = pd.concat(Etestdf).reset_index(drop = True)
            Edf = Edf.query('Year != 2020 and Year != 2021')
            Edf = Edf.reset_index(drop = True)  
            del Edf['Year']                 
                  
        Etrain, Eval = Edf.loc[trainIdx], Edf.loc[valIdx]   
        Etest = Etestdf.copy()
        
    #break   
        
    # concat dataframes and bind target
    #print(ytrain)        
    xtrain = ytrain.copy()
    #print(xtrain)
    xval = yval.copy()
    xtest = ytest.copy()
    gc.collect() 
    if args.A or args.D or args.AEI or args.DEI:
        if args.A or args.D:
            K = pd.concat(kinships, axis=1)
            xtrain = pd.merge(xtrain, K, on='Hybrid', how='left').dropna()#.set_index(['Env', 'Hybrid'])
            xval = pd.merge(xval, K, on='Hybrid', how='left').dropna()#.set_index(['Env', 'Hybrid'])
            xtest = pd.merge(xtest, K, on='Hybrid', how='left')#.set_index(['Env', 'Hybrid']) #remove dropna()
            del kinships
            gc.collect()
        if args.AEI or args.DEI:
            kron = pd.concat(kroneckers, axis=1)
            del kroneckers
            xtrain = pd.merge(xtrain, kron, on=['Env', 'Hybrid'], how='inner')
            xval = pd.merge(xval, kron, on=['Env', 'Hybrid'], how='inner')
            xtest = pd.merge(xtest, kron, on=['Env', 'Hybrid'], how='inner')
            del kron 
            gc.collect() 

    # split x, y 
    ytrain = xtrain[['Yield_Mg_ha', 'Env', 'Hybrid']] 
    del xtrain['Yield_Mg_ha']
    yval = xval[['Yield_Mg_ha', 'Env', 'Hybrid']]
    del xval['Yield_Mg_ha']
    ytest = xtest[['Yield_Mg_ha', 'Env', 'Hybrid']]
    del xtest['Yield_Mg_ha']    
    gc.collect()

    # bind lagged yield features
    no_lags_cols = [x for x in xtrain.columns.tolist() if x not in ['Env', 'Hybrid']]
    #print(no_lags_cols[0])
    if args.lag_features:
        if cv_count == 1:
            outfile = f'{outfile}_lag_features'
        lagdf = pd.read_csv(OUTPUT_PATH / f'gxe_xtrain.csv', usecols=lambda x: 'yield_lag' in x or x in ['Env', 'Hybrid']).set_index(['Env', 'Hybrid'])
        if args.drop:
            lagdf['Year'] = lagdf['Env'].str[-4:].astype('int')
            lagdf = lagdf.query('Year != 2018 and Year != 2019')
            del lagdf['Year']
        xtrain_lag, xval_lag, xtest_lag = lagdf[trainIdx], lagdf[valIdx], lagdf[testIdx]
        xtrain = xtrain.merge(xtrain_lag, on=['Env', 'Hybrid'], how='inner').copy()
        xval = xval.merge(xval_lag, on=['Env', 'Hybrid'], how='inner').copy()
        xtest = xtest.merge(xtest_lag, on=['Env', 'Hybrid'], how='inner').copy()
    else:
        xtrain_id = xtrain[['Env', 'Hybrid']].copy()
        xval_id = xval[['Env', 'Hybrid']].copy()
        xtest_id = xtest[['Env', 'Hybrid']].copy()        

    """
    if args.AEI or args.DEI:
        if 'Env' in xtrain.columns and 'Hybrid' in xtrain.columns:
            xtrain = xtrain.set_index(['Env', 'Hybrid'])
            xval = xval.set_index(['Env', 'Hybrid'])
            xtest = xtest.set_index(['Env', 'Hybrid']) 
    """                 
        
    #break        

    # run model
    if not args.svd:

        # add factor     
        xtrain = xtrain.reset_index(drop=True)
        xtrain = create_field_location(xtrain)
        xtrain['Field_Location'] = xtrain['Field_Location'].astype('category')
        xtrain = create_state_location(xtrain)
        xtrain['State_Location'] = xtrain['State_Location'].astype('category')
        xtrain = xtrain.set_index(['Env', 'Hybrid'])
        xval = xval.reset_index(drop=True)
        xval = create_field_location(xval)
        xval['Field_Location'] = xval['Field_Location'].astype('category')
        xval = create_state_location(xval)
        xval['State_Location'] = xval['State_Location'].astype('category')        
        xval = xval.set_index(['Env', 'Hybrid'])
        xtest = xtest.reset_index(drop=True)
        xtest = create_field_location(xtest)
        xtest['Field_Location'] = xtest['Field_Location'].astype('category')
        xtest = create_state_location(xtest)
        xtest['State_Location'] = xtest['State_Location'].astype('category')        
        xtest = xtest.set_index(['Env', 'Hybrid'])       
     
        ytrain = ytrain.reset_index(drop=True)
        ytrain = create_field_location(ytrain)
        ytrain['Field_Location'] = ytrain['Field_Location'].astype('category')
        ytrain = create_state_location(ytrain)
        ytrain['State_Location'] = ytrain['State_Location'].astype('category')        
        ytrain = ytrain.set_index(['Env', 'Hybrid'])
        
        yval = yval.reset_index(drop=True)
        yval = create_field_location(yval)
        yval['Field_Location'] = yval['Field_Location'].astype('category')
        yval = create_state_location(yval)
        yval['State_Location'] = yval['State_Location'].astype('category')        
        yval = yval.set_index(['Env', 'Hybrid'])   
        
        ytest = ytest.reset_index(drop=True)
        ytest = create_field_location(ytest)
        ytest['Field_Location'] = ytest['Field_Location'].astype('category')
        ytest = create_state_location(ytest)
        ytest['State_Location'] = ytest['State_Location'].astype('category')        
        ytest = ytest.set_index(['Env', 'Hybrid'])                
        
        #break 

        # include E matrix if requested (Notice that this matrix was built using SVD, therefore don't apply SVD it again)
        if args.E:
            lag_cols = xtrain.filter(regex='_lag', axis=1).columns
            if len(lag_cols) > 0:
                xtrain = xtrain.drop(lag_cols, axis=1)
                xval = xval.drop(lag_cols, axis=1)
                xtest = xtest.drop(lag_cols, axis=1)
            xtrain = xtrain.merge(Etrain, on=['Env', 'Hybrid'], how='left').set_index(['Env', 'Hybrid'])
            xval = xval.merge(Eval, on=['Env', 'Hybrid'], how='left').set_index(['Env', 'Hybrid'])
            xtest = xtest.merge(Etest, on=['Env', 'Hybrid'], how='left').set_index(['Env', 'Hybrid'])
            del Etrain, Eval, Etest
            gc.collect()

        # break
        xtrain = xtrain.query('Field_Location != "GEH1" and Field_Location != "COH1" and Field_Location != "ARH1" and Field_Location != "ARH2"')
        ytrain = ytrain.query('Field_Location != "GEH1" and Field_Location != "COH1" and Field_Location != "ARH1" and Field_Location != "ARH2"')
        del xtrain['Field_Location']
        del ytrain['Field_Location']
        ytrain.reset_index()
        ytrain = ytrain['Yield_Mg_ha']
		 	
        print('Using full set of features.')
        print('# Features:', xtrain.shape[1] - 2)
               
        # fit
        model = lgbm.LGBMRegressor(random_state=args.seed, max_depth=6, num_leaves = 8)
        model.fit(xtrain, ytrain)
        
    else:
        if cv_count == 1:
            outfile = f'{outfile}_svd{args.n_components}comps'
        print('Using svd.')
        print('# Components:', args.n_components)
     
        #break   
        
        #print(no_lags_cols[0])                
        svd = TruncatedSVD(n_components=args.n_components, random_state=args.seed)
        svd.fit(xtrain[no_lags_cols])  # fit but without lagged yield features
        print('Explained variance:', svd.explained_variance_ratio_.sum())

        # transform from the fitted svd
        svd_cols = [f'svd{i}' for i in range(args.n_components)]
        #xtrain_svd = pd.DataFrame(svd.transform(xtrain[no_lags_cols]), columns=svd_cols, index=xtrain[no_lags_cols].index)
        xtrain_svd = pd.DataFrame(svd.transform(xtrain[no_lags_cols]), columns=svd_cols, index=xtrain[['Env', 'Hybrid']].index)
        xval_svd = pd.DataFrame(svd.transform(xval[no_lags_cols]), columns=svd_cols, index=xval[['Env', 'Hybrid']].index)
        xtest_svd = pd.DataFrame(svd.transform(xtest[no_lags_cols]), columns=svd_cols, index=xtest[['Env', 'Hybrid']].index)
        del svd
        gc.collect() 

        # bind lagged yield features if needed
        if args.lag_features:
            xtrain_lag = xtrain_lag.set_index(['Env', 'Hybrid'])
            xtrain = xtrain_svd.merge(xtrain_lag, on=['Env', 'Hybrid'], how='inner').copy()
            del xtrain_svd, xtrain_lag
            xval_lag = xval_lag.set_index(['Env', 'Hybrid'])
            xval = xval_svd.merge(xval_lag, on=['Env', 'Hybrid'], how='inner').copy()
            del xval_svd, xval_lag
            xtest_lag = xtest_lag.set_index(['Env', 'Hybrid'])
            xtest = xtest_svd.merge(xtest_lag, on=['Env', 'Hybrid'], how='inner').copy()
            del xtest_svd, xtest_lag            
            gc.collect()
        else:                   
            xtrain_id.reset_index(inplace=True)
            xtrain_svd.reset_index(inplace=True)
            xtrain = xtrain_svd.merge(xtrain_id, on = ['index'], how = 'inner').copy()
            del xtrain_svd, xtrain_id
            del xtrain['index']  
            xval_id.reset_index(inplace=True)
            xval_svd.reset_index(inplace=True)
            xval = xval_svd.merge(xval_id, on = ['index'], how = 'inner').copy()
            del xval_svd, xval_id
            del xval['index']  
            xtest_id.reset_index(inplace=True)
            xtest_svd.reset_index(inplace=True)
            xtest = xtest_svd.merge(xtest_id, on = ['index'], how = 'inner').copy()
            del xtest_svd, xtest_id  
            del xtest['index']           
            gc.collect()
                 
        # add factor
        xtrain = xtrain.reset_index(drop=True)
        xtrain = create_field_location(xtrain)
        xtrain['Field_Location'] = xtrain['Field_Location'].astype('category')
        xtrain = create_state_location(xtrain)
        xtrain['State_Location'] = xtrain['State_Location'].astype('category')
        xtrain = xtrain.set_index(['Env', 'Hybrid'])
        xval = xval.reset_index(drop=True)
        xval = create_field_location(xval)
        xval['Field_Location'] = xval['Field_Location'].astype('category')
        xval = create_state_location(xval)
        xval['State_Location'] = xval['State_Location'].astype('category')        
        xval = xval.set_index(['Env', 'Hybrid'])
        xtest = xtest.reset_index(drop=True)
        xtest = create_field_location(xtest)
        xtest['Field_Location'] = xtest['Field_Location'].astype('category')
        xtest = create_state_location(xtest)
        xtest['State_Location'] = xtest['State_Location'].astype('category')        
        xtest = xtest.set_index(['Env', 'Hybrid'])       
     
        ytrain = ytrain.reset_index(drop=True)
        ytrain = create_field_location(ytrain)
        ytrain['Field_Location'] = ytrain['Field_Location'].astype('category')
        ytrain = create_state_location(ytrain)
        ytrain['State_Location'] = ytrain['State_Location'].astype('category')        
        ytrain = ytrain.set_index(['Env', 'Hybrid'])
        
        yval = yval.reset_index(drop=True)
        yval = create_field_location(yval)
        yval['Field_Location'] = yval['Field_Location'].astype('category')
        yval = create_state_location(yval)
        yval['State_Location'] = yval['State_Location'].astype('category')        
        yval = yval.set_index(['Env', 'Hybrid'])   
        
        ytest = ytest.reset_index(drop=True)
        ytest = create_field_location(ytest)
        ytest['Field_Location'] = ytest['Field_Location'].astype('category')
        ytest = create_state_location(ytest)
        ytest['State_Location'] = ytest['State_Location'].astype('category')        
        ytest = ytest.set_index(['Env', 'Hybrid'])                
                
        # include E matrix if requested (Notice that this matrix was built using SVD, therefore don't apply SVD it again)
        if args.E:
            xtrain = xtrain.merge(Etrain, on=['Env', 'Hybrid'], how='left').copy().set_index(['Env', 'Hybrid'])
            xval = xval.merge(Eval, on=['Env', 'Hybrid'], how='left').copy().set_index(['Env', 'Hybrid'])
            xtest = xtest.merge(Etest, on=['Env', 'Hybrid'], how='left').copy().set_index(['Env', 'Hybrid'])
            lag_cols = xtrain.filter(regex='_lag', axis=1).columns
            if len(lag_cols) > 0:
                xtrain = xtrain.drop(lag_cols, axis=1)
                xval = xval.drop(lag_cols, axis=1)
                xtest = xtest.drop(lag_cols, axis=1)     
            
        print('Using SVD set of features.')
        print('# Features:', xtrain.shape[1])                          

        # break    
        xtrain = xtrain.query('Field_Location != "GEH1" and Field_Location != "COH1" and Field_Location != "ARH1" and Field_Location != "ARH2"')
        ytrain = ytrain.query('Field_Location != "GEH1" and Field_Location != "COH1" and Field_Location != "ARH1" and Field_Location != "ARH2"')
        del xtrain['Field_Location']
        del ytrain['Field_Location']        
        ytrain.reset_index()
        ytrain = ytrain['Yield_Mg_ha']
			
        model = lgbm.LGBMRegressor(random_state=args.seed, max_depth=6, num_leaves = 8)
        model.fit(xtrain, ytrain)#, eval_metric = train_eval)
     
    xval = xval.query('Field_Location != "GEH1" and Field_Location != "COH1" and Field_Location != "ARH1" and Field_Location != "ARH2"')
    yval = yval.query('Field_Location != "GEH1" and Field_Location != "COH1" and Field_Location != "ARH1" and Field_Location != "ARH2"')            
    yval.reset_index(inplace=True, drop=True)
    yval = yval['Yield_Mg_ha']
    ytest.reset_index(inplace=True, drop=True)
    ytest = ytest['Yield_Mg_ha']
    
    del xval['Field_Location']
    del xtest['Field_Location']

    # predict
    ypred_train = model.predict(xtrain)
    ypred_val = model.predict(xval)
    ypred_test = model.predict(xtest)

    # validate
    df_eval_train = create_df_eval(xtrain, ytrain, ypred_train)
    df_eval_val = create_df_eval(xval, yval, ypred_val)   
    print("RMSE eval", avg_rmse(df_eval_val))       
    df_eval_test = create_df_eval(xtest, ytest, ypred_test)

    # feature importance
    df_feat_imp = feat_imp(model)
    if cv_count == 1:
        feat_imp_outfile = f'{outfile.replace("oof", "feat_imp")}.csv'
        df_feat_imp.to_csv(feat_imp_outfile, index=False)
  
    #df_eval_test['Validation'] = valYear
    concat_val_pred.append(df_eval_val)
    concat_test_pred.append(df_eval_test)
          
    cv_count = cv_count + 1          
    #if cv_count > 1:
    #    break

#print(cv_count)
concat_val_pred = pd.concat(concat_val_pred)
concat_test_pred = pd.concat(concat_test_pred)

# write OOF results
outfile_val = f'{outfile}_val.csv'
outfile_test = f'{outfile}_test.csv'
print(outfile_val)
print(outfile_test)

concat_val_pred.to_csv(outfile_val, index=False)
concat_test_pred.to_csv(outfile_test, index=False)
