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
import os
import sys

#%% for evaluation
def ven2(lia, lib):
    lic = list(set(lia).intersection(lib))
    print('A: {}\nB: {}\noverlap:{}'.format(len(lia), len(lib), len(lic)))

uniq = lambda x: sorted(list(set(x.values.tolist())))

rmse = lambda predictions,targets: np.sqrt(np.mean((np.array(predictions)-np.array(targets))**2))

cp = lambda x: copy.deepcopy(x)

def evaluatepred(pred, gt):
    plotit = True
    env2pr = {}
    env2rm = {}
    all_pred, all_gt = [], []
    # use pred as main data points
    envs = pred['Env'].unique()
    mi_x, ma_x = 100,0
    mi_y, ma_y = 100,0
    for env in envs:
        print('-- env:', env, '--', end='')
        env_pred = pred[pred['Env'] == env]
        env_n    = len(env_pred) # num of hybrids to be predicted in this env
        env_gt   = gt[gt['Env'] == env]
        env_join = pd.merge(env_pred, env_gt, on = 'Hybrid')
        env_n_jo = len(env_join)
        print('hit {}/{} ({:.2f})'.format(env_n_jo, env_n, env_n_jo/env_n))
        if env_n_jo == 0: # env no hit
            continue
        env_join.dropna(subset=['Yield_Mg_ha_y'], inplace=True)
        v_pred = env_join['Yield_Mg_ha_x'].values
        v_gt   = env_join['Yield_Mg_ha_y'].values
        # v_gt   = v_gt - np.mean(v_gt)
        all_pred += v_pred.tolist()
        all_gt   += v_gt.tolist()
        env2pr[env] = pearsonr(v_pred, v_gt)[0]
        env2rm[env] = rmse(v_pred, v_gt)
        print('pr: {:.3f}, rmse: {:.3f}'.format(env2pr[env], env2rm[env]))
        mi_x = np.min(v_pred) if np.min(v_pred) < mi_x else mi_x
        ma_x = np.max(v_pred) if np.max(v_pred) > ma_x else ma_x
        mi_y = np.min(v_gt) if np.min(v_gt) < mi_y else mi_y
        ma_y = np.max(v_gt) if np.max(v_gt) > ma_y else ma_y
        if plotit:
            plt.scatter(v_pred, v_gt, s=15, alpha=0.46)
    ave_pr = np.mean([v for k,v in env2pr.items()]) # averaged r
    ave_rm = np.mean([v for k,v in env2rm.items()]) # averaged rmse
    ave_pr_acr = pearsonr(all_pred, all_gt)[0] # across-env r
    ave_rm_acr = rmse(all_pred, all_gt) # across-env rmse
    plt.xlabel('predicted')
    plt.ylabel('truth yield')
    plt.title('ave pr: {:.4f}^ ({:.3f}v), across pr: {:.3f} ({:.3f})'.format(ave_pr,
                            ave_rm, ave_pr_acr, ave_rm_acr))
    plt.plot([0,17],[0,17],'k:')
    smv = 0.5
    plt.xlim([mi_x-smv, ma_x+smv])
    plt.ylim([mi_y-smv, ma_y+smv])
    return env2pr

#%%
def evaluatepoch(pred, gt):
    env2pr = []    
    # use pred as main data points
    envs = pred['Env'].unique()    
    for ev in envs:
        # print('-- env:', ev, '--', end='')
        env_pred = pred[pred['Env'] == ev]
        env_n    = len(env_pred) # num of hybrids to be predicted in this env
        env_gt   = gt[gt['Env'] == ev]
        env_join = pd.merge(env_pred, env_gt, on = 'Hybrid')
        env_n_jo = len(env_join)
        
        if env_n_jo == 0: # env no hit
            continue
        env_join.dropna(subset=['Yield_Mg_ha_y'], inplace=True)
        v_pred = env_join['Yield_Mg_ha_x'].values
        v_gt   = env_join['Yield_Mg_ha_y'].values
        
        _pr_   = pearsonr(v_pred, v_gt)[0]
        # print('hit {}/{} ({:.2f}) - {:.4f}'.format(env_n_jo, env_n, env_n_jo/env_n, _pr_))
        env2pr.append(_pr_)                
    # ave_pr = np.mean([v for k,v in env2pr.items()]) # averaged r    
    return env2pr

def evaluatepoch_(pred, gt): # get env names
    env2pr = []    
    # use pred as main data points
    envs = pred['Env'].unique()    
    for env in envs:
        # print('-- env:', env, '--', end='')
        env_pred = pred[pred['Env'] == env]
        env_n    = len(env_pred) # num of hybrids to be predicted in this env
        env_gt   = gt[gt['Env'] == env]
        env_join = pd.merge(env_pred, env_gt, on = 'Hybrid')
        env_n_jo = len(env_join)
        # print('hit {}/{} ({:.2f})'.format(env_n_jo, env_n, env_n_jo/env_n))
        if env_n_jo == 0: # env no hit
            continue
        env2pr.append(env)
    return env2pr

def mergeback(allp, end):
    nepo = end # allp.shape[0]
    nllp = np.zeros_like(allp)[:end]
    for i in range(nepo):
        nllp[nepo-(1+i),:] = np.mean(allp[end-(i+1):end,:], axis=0) # !!bug: not [-(i+1):,:]
    return nllp
#%% for feature preprocessing
getenv = lambda df_main: set(df_main['Env'].tolist())

def checkclean(dftr_meta, fea_meta): 
    for fea in fea_meta:
        allv = dftr_meta[fea]
        nas = allv.isna().sum()
        # only for categorical, remove nan, sort, then see them
        uniqv = None if allv.dtype=='float64' else allv.dropna().values.tolist()
        occ = {}
        if uniqv != None: # not numerical
            uniqvs = set(uniqv)
            for k in uniqvs:
                occ[k] = uniqv.count(k)
            uniqvs = sorted(uniqvs)
        print('\n{} - nan: {}/{}, unique: {}'.format(fea, nas, len(allv), len(occ)))
        if uniqv != None:
            print([str(x)+':'+str(occ[x]) for x in uniqvs])

def getdata(dftr_meta, fea_meta): 
    '''
    dftr_meta - complete datatable
    fea_meta - selected feature list
    '''
    _key_ = dftr_meta['Env'].values
    _val_ = dftr_meta[fea_meta].values
    return dict(zip(_key_,_val_))

def save_env2fea(data_meta,fname):
    with open(fname, "wb") as f:
        pickle.dump(data_meta, f)

path = lambda di,fi: os.path.join(di,fi)

#%% feature imputation

near_env = lambda query,data_so:[env for env in list(data_so.keys()) if env.startswith(query[:4])]

def sel_2(env, n_env): # select 2 envs from the list
    N = 2
    if len(n_env) < 3:
        return n_env
    else:
        _y_  = int(env[-4:])
        _ys_ = [abs(_y_ - int(i[-4:])) for i in n_env]
        idx  = np.argsort(_ys_)
        return np.array(n_env)[idx][:N]

dim_soil = 24
def sel_mean(sel, data_so):
    vs = [data_so[env] for env in sel]
    vs = np.array(vs)
    if vs.shape[1] == dim_soil: # soil data only!
        vs = vs[:,:-1]
        vs = vs.astype(float)
    vs[np.isnan(vs)] = 0 # soil data has lots of missing numbers
    return np.mean(vs, axis=0)

def data_impu(train_envs, data_so):
    impu = {}
    for env in train_envs:
        if env in data_so:
            continue
        # env = 'TXH1-Early_2018'
        n_env = near_env(env, data_so)
        if len(n_env) == 0:
            continue
        _sel_ = sel_2(env, n_env)
        print(env, _sel_)
        _imp_ = sel_mean(_sel_, data_so)
        _imp_ = _imp_.tolist()
        if len(_imp_) == dim_soil - 1: # soil data only!
            _imp_.append(np.nan) # note format here
        impu[env] = np.array(_imp_, dtype=float)
    return impu

getdim = lambda data_meta : len(next(iter(data_meta.values())))

def assimpu(data_soil, data_soil_impu):
    assert getdim(data_soil) == getdim(data_soil_impu)    

#%%
def save2dl(fn, x):
    print('saving {} to {} ({:.3f} ~ {:.3f}):'.format(x.shape, fn, x.min(), x.max()))
    np.save(fn, x)

def savejs(fn, d):
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)

#%% distribution calibration
import statsmodels.api as sm

def plotqq(_y_, ax1, env, usefit=False, c='b', alpha=0.2):
    pp = sm.ProbPlot(_y_, fit=usefit)
    qq = pp.qqplot(marker='.', 
                   markerfacecolor=c, markeredgecolor=c, 
                   alpha=alpha, ax=ax1, label=env)

def plotqq2(_x_, _y_, ax1, env, usefit=False):    
    pp_x = sm.ProbPlot(_x_, fit=usefit)
    pp_y = sm.ProbPlot(_y_, fit=usefit)
    sm.qqplot_2samples(pp_x, pp_y, ax=ax1, line='45') # line='45'

def quantile_mapping_(predictions, target):
    # here two arrays must have SAME length
    pred_sorted = np.sort(predictions)
    target_sorted = np.sort(target)
    
    # Map prediction quantiles to target quantiles
    mapped_predictions = np.interp(predictions, pred_sorted, target_sorted)
    return mapped_predictions

def quantile_mapping(predictions, target): # allows diff lengths
    # Sort the predictions and target
    pred_sorted = np.sort(predictions)
    target_sorted = np.sort(target)
    
    # Generate cumulative distribution indices for predictions and targets
    pred_cdf = np.linspace(0, 1, len(pred_sorted), endpoint=True)
    target_cdf = np.linspace(0, 1, len(target_sorted), endpoint=True)
    
    # Interpolate prediction values into the target CDF
    mapped_predictions = np.interp(
        np.interp(predictions, pred_sorted, pred_cdf),  # Map predictions to their CDF values
        target_cdf,  # Use the target CDF as the "x" axis
        target_sorted  # Map to the target values
    )
    return mapped_predictions

def histogram_matching_(predictions, target):
    # here two arrays must have SAME length
    pred_values, pred_bins = np.histogram(predictions, bins=100, density=True)
    target_values, target_bins = np.histogram(target, bins=100, density=True)
    
    pred_cdf = np.cumsum(pred_values)
    target_cdf = np.cumsum(target_values)
    
    pred_cdf /= pred_cdf[-1]
    target_cdf /= target_cdf[-1]
    
    mapping = np.interp(predictions, pred_bins[:-1], pred_cdf)
    calibrated_predictions = np.interp(mapping, target_cdf, target_bins[:-1])
    return calibrated_predictions

def histogram_matching(predictions, target, bins=100):
    # Compute histograms and CDFs
    pred_values, pred_bins = np.histogram(predictions, bins=bins, density=True)
    target_values, target_bins = np.histogram(target, bins=bins, density=True)
    
    pred_cdf = np.cumsum(pred_values)
    target_cdf = np.cumsum(target_values)
    
    # Normalize CDFs
    pred_cdf /= pred_cdf[-1]
    target_cdf /= target_cdf[-1]
    
    # Map prediction values to target's histogram
    calibrated_predictions = np.interp(predictions, pred_bins[:-1], np.interp(pred_cdf, pred_cdf, target_bins[:-1]))
    return calibrated_predictions
    
#%%
def quantile_binning(arr, n_bins):
    """
    Perform quantile binning on a numpy array to create n_bins with equal number of samples.
    
    Parameters:
    arr (numpy.ndarray): Input array to be binned
    n_bins (int): Number of bins desired
    
    Returns:
    tuple:
        - bin_edges (numpy.ndarray): Array of bin boundaries (length n_bins + 1)
        - bin_indices (numpy.ndarray): Array of same shape as input with bin indices (0 to n_bins-1)
        - bin_counts (numpy.ndarray): Number of samples in each bin
    """
    # Calculate quantiles for bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(arr, quantiles)
    
    # Ensure the last bin edge is slightly larger than max value
    bin_edges[-1] = bin_edges[-1] * (1 + 1e-10)
    
    # Assign each value to a bin
    bin_indices = np.digitize(arr, bin_edges[1:-1])
    
    # Count samples in each bin
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    return bin_edges, bin_indices, bin_counts

def binning(arr, n_bins):
    """
    Perform constant width binning on a numpy array.
    
    Parameters:
    arr (numpy.ndarray): Input array to be binned
    n_bins (int): Number of bins desired
    
    Returns:
    tuple:
        - bin_edges (numpy.ndarray): Array of bin boundaries (length n_bins + 1)
        - bin_indices (numpy.ndarray): Array of same shape as input with bin indices (0 to n_bins-1)
        - bin_counts (numpy.ndarray): Number of samples in each bin
    """
    # Calculate bin edges with constant width
    min_val = np.min(arr)
    max_val = np.max(arr)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # Ensure the last bin edge is slightly larger than max value
    bin_edges[-1] = bin_edges[-1] * (1 + 1e-10)
    
    # Assign each value to a bin
    bin_indices = np.digitize(arr, bin_edges[1:-1])
    
    # Count samples in each bin
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    return bin_edges, bin_indices, bin_counts

#%% error analysis
def res_stat(t_envs, df_, method): # statistics of results
    '''
    note: this can only give an estimated testing pr! because dummy values
    '''
    x_names = []
    y  = []
    ys = []
    name2dist = {}
    nu = []
    rs = []
    mm = []
    me = []
    
    for env in t_envs:
        _d_ = df_[df_['Env']==env]
        print('{} in train-set:{}'.format(env,len(_d_)))
        nu.append(len(_d_))
        # _v_ = _d_['Yield_Mg_ha'].dropna().values
        _v_ = _d_['Yield_Mg_ha'].values # shouldn't have nan # real values
        _p_ = _d_[method].values # predicted values
        mm.append(_v_.max() - _v_.min())
        _r_ = pearsonr(_p_, _v_)[0]
        rs.append(_r_)
        _y_ = np.mean(_v_)
        q1 = np.percentile(_v_, 25)
        q3 = np.percentile(_v_, 75)
        iqr = q3 - q1
        _s_ = np.std(_v_)
        x_names.append(env)
        name2dist[env] = _v_
        y.append(_y_)
        # me.append(np.median(_v_))
        me.append(iqr)
        ys.append(_s_)
    
    d_f = pd.DataFrame.from_dict({'Env': x_names, 'hybrids': nu,
                                  'meany': y, 'stdy': ys, 'mmy': mm,'pr': rs})
    return d_f

def standardize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
#%%
from scipy.special import softmax
from tqdm import tqdm

def read_epoch_dir(res_dir, df_tar, nepo, topn, binmean, gt):
    '''
    res_dir: result dir
    df_tar: read submission file
    nepo: 24
    topn: 15
    binmean: read from pkl
    '''
    pred = df_tar.copy(deep=True)
    mprs = []
    allp = [] # all env-hyb predictions
    
    ## nepo = 24
    for i in tqdm(range(nepo)):
        predmlp = np.load(path(res_dir, 'pred_'+str(i)+'.npy'))
        if predmlp.ndim == 2:
            # predmlp = np.argmax(predmlp, axis=1)    
            predmlp = softmax(predmlp, axis=1) 
            maxs = np.argsort(predmlp, axis=1)[:,-topn:] # 15
            prds = np.sort(predmlp, axis=1)[:,-topn:] # 15
            predmlp = np.sum(binmean[maxs]*prds, axis=1) / np.sum(prds, axis=1)
        
        allp.append(predmlp.tolist())
        ## assert len(predmlp) == len_va 
        pred['Yield_Mg_ha'] = predmlp # + mean_yie
        _pr_ = evaluatepoch(pred, gt)
        mprs.append(_pr_)
    resul = np.array(mprs)
    allp  = np.array(allp)
    mpr   = np.mean(resul, axis=1)
    # print('epoch x envs:', resul.shape)
    mpi   = np.argmax(mpr)
    end_epo = nepo
    return resul, allp, mpr, mpi

def scoreINbin(predmlp, topn, binmean):
    assert predmlp.ndim == 2
    predmlp = softmax(predmlp, axis=1) 
    maxs = np.argsort(predmlp, axis=1)[:,-topn:] # 15
    prds = np.sort(predmlp, axis=1)[:,-topn:] # 15
    predmlp = np.sum(binmean[maxs]*prds, axis=1) / np.sum(prds, axis=1)
    return predmlp

#%%
def plot_mean_prs(resul, mpr, mpi, nepo=24):
    fig = plt.figure()
    plt.plot(resul, color='#BFB5D7')
    plt.plot(mpr, color='r', marker="^", linestyle='None') 
    plt.grid()
    for i in range(nepo):
        plt.text(i, mpr[i], '{:.4f}'.format(mpr[i]), fontsize=9, rotation=-45)
    plt.title('best epoch: {}:{:.4f} (last {:.4f})'.format(mpi+1, mpr[mpi], mpr[-1]))
    
#%%
def binlab(yori, outfi, bin_edges):
    _n_bins = len(bin_edges) - 1
    print('original label range:', yori.min(), yori.max())
    y_bin = np.digitize(yori, bin_edges[1:-1])
    bin_counts = np.bincount(y_bin, minlength=_n_bins)    
    print('binned label range:', y_bin.min(), y_bin.max())
    for i in range(_n_bins):
        print('{:.2f} - {:.2f}: {}'.format(bin_edges[i], 
                                           bin_edges[i+1], 
                                           bin_counts[i]))
    np.save(outfi, y_bin)

#%% eigenvalues
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

def pairwise_distance_with_nan(X):
    """
    Compute pairwise distances handling NaN values.
    For each pair of samples, only use features that are non-NaN in both.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input matrix that may contain NaN values
    
    Returns:
    --------
    distances : array of shape (n_samples * (n_samples-1) / 2,)
        Condensed distance matrix
    
    input:
        X = geno24.values.T
    not use, too slow
    """
    n_samples = X.shape[0]
    distances = np.zeros(n_samples * (n_samples - 1) // 2)
    k = 0
    
    for i in tqdm(range(n_samples)):
        for j in range(i + 1, n_samples):
            # Get mask of features that are non-NaN in both samples
            valid_mask = ~np.isnan(X[i, :]) & ~np.isnan(X[j, :])
            
            if np.sum(valid_mask) == 0:
                # If no valid features, use maximum possible distance
                distances[k] = np.sqrt(np.sum(valid_mask))  # or another default value
            else:
                # Compute Euclidean distance using only valid features
                diff = X[i, valid_mask] - X[j, valid_mask]
                # Normalize by number of valid features to make distances comparable
                distances[k] = np.sqrt(np.sum(diff ** 2) * (X.shape[1] / np.sum(valid_mask)))
            k += 1
    
    return distances

def compute_eigenfeatures(X, n_components=None, missing_threshold=0.5):
    """
    Compute eigenfeatures from a feature matrix using distance-based approach,
    handling missing values.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input feature matrix (e.g., SNP data)
    n_components : int, optional
        Number of top eigenvalues/vectors to return. If None, returns all
    missing_threshold : float, optional
        Maximum fraction of missing values allowed per feature/sample
        Features or samples with more missing values will be removed
        
    Returns:
    --------
    eigenvalues : array of shape (n_components,)
        The top eigenvalues
    eigenvectors : array of shape (n_samples, n_components)
        The corresponding eigenvectors
    transformed_X : array of shape (n_samples, n_components)
        The projection of X onto the eigenvectors
    """
    X = np.array(X, dtype=float)  # Ensure float type for NaN handling
    
    # Remove features with too many missing values
    # feature_missing_rate = np.isnan(X).mean(axis=0)
    # good_features = feature_missing_rate < missing_threshold
    # X = X[:, good_features]
    
    # Remove samples with too many missing values
    # sample_missing_rate = np.isnan(X).mean(axis=1)
    # good_samples = sample_missing_rate < missing_threshold
    # X = X[good_samples]
    
    # Step 1: Handle missing values in standardization
    '''
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    X_scaled = (X - means) / stds
    '''
    X_scaled = X
    # Step 2: Compute pairwise distances handling NaN values
    # distances = pairwise_distance_with_nan(X_scaled)
    distances = pdist(X_scaled)
    
    # Step 3: Convert distance vector to square matrix
    distance_matrix = squareform(distances)
    
    # Step 4: Center the kernel matrix
    n = distance_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = -0.5 * H.dot(distance_matrix).dot(H)  # Double centering
    
    # Step 5: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(K)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components if specified
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    # Project data onto eigenvectors
    transformed_X = np.dot(K, eigenvectors)
    
    print(f"Original data shape: {X.shape}")
    print(f"Eigenvalues shape: {eigenvalues.shape}")
    print(f"Eigenvectors shape: {eigenvectors.shape}")
    print(f"Transformed data shape: {transformed_X.shape}")
    print("\nTop 5 eigenvalues:", eigenvalues[:5])
    
    return eigenvalues, eigenvectors, transformed_X, distances

#%% ========== ========== ========== ========== ========== ==========
# catboost
from catboost import CatBoostRegressor, Pool, metrics, cv
from catboost import EShapCalcType, EFeaturesSelectionAlgorithm

def select_features_syntetic(train_pool: Pool,
                             test_pool: Pool,
                             algorithm: EFeaturesSelectionAlgorithm,                              
                             loss: str = 'RMSE',
                             raw_dim: int = 13328,
                             select: int = 300,
                             steps: int = 1):
    print('Algorithm:', algorithm)
    print('loss:', loss)
    print('raw_dim:', raw_dim)
    print('selected dim:', select)
    model = CatBoostRegressor(iterations = 600, 
                              task_type = 'CPU',
                              loss_function = loss,
                              # loss_function = 'Quantile:alpha=0.75',
                              random_seed = 0,
                              early_stopping_rounds = 50)
    summary = model.select_features(
        train_pool,
        eval_set = test_pool,
        # we will select from all features
        features_for_select = list(range(raw_dim)), # X.shape[1]
        # we want to select exactly important features
        num_features_to_select = select,  
        # more steps - more accurate selection
        steps = steps,                                     
        algorithm = algorithm,
        # can be Approximate, Regular and Exact
        shap_calc_type = EShapCalcType.Regular,            
        # to train model with selected features
        train_final_model = True,                          
        logging_level = 'Verbose',
        plot = True
    )
    print(summary['selected_features'])
    return summary

#%%
def plot45(ax):
    # Get the axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Calculate the coordinates for the 45-degree line
    x_line = [min(x_min, y_min), max(x_max, y_max)]
    y_line = [min(x_min, y_min), max(x_max, y_max)]
    
    # Plot the 45-degree line
    ax.plot(x_line, y_line, color='red', linestyle='--', alpha=0.5)
#%% ========== ========== ========== ========== ========== ==========

from torch.utils.data import Dataset, DataLoader
class tabDataset(Dataset):    
    '''
    we need --
    data e.g.: torch.Size([10000, 28, 28])
    targets    
    '''
    def __init__(self, data, labels, reg=False):
        '''
        we generate several classes
        '''       
                
        self.data = torch.tensor(data, dtype=torch.float32)
        if reg:
            self.targets = torch.tensor(labels, dtype=torch.float32)
        else:
            self.targets = torch.tensor(labels, dtype=torch.long)
        
        # here is the reason for dtype
        # ref: https://discuss.pytorch.org/t/dataloader-overrides-tensor-type/9861
        # Because your target is 1-dimensional, Pytorch casts the elements to DoubleTensors.
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ''' what's the usage of this?
        if torch.is_tensor(idx):
            idx = idx.tolist()
        '''
        return (self.data[idx], self.targets[idx])

class mixDataset(Dataset):    
    def __init__(self, data, labels, IDs, reg=False):  
        self.data = torch.tensor(data, dtype=torch.float32)
        if reg:
            self.targets = torch.tensor(labels, dtype=torch.float32)
        else:
            self.targets = torch.tensor(labels, dtype=torch.long)
        self.IDs = IDs
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data[idx], self.targets[idx], self.IDs[idx])
    
#%%
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class mlp_1l(nn.Module):
    '''
    mlp: 1 hidden layer
    ref: https://github.com/Duncanswilson/maxout-pytorch/blob/master/maxout_pytorch.ipynb
    BatchNorm: https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989
    '''
    def __init__(self, num_inputs, 
                       num_hiddens, 
                       num_outputs):
        super(mlp_1l, self).__init__()
        self.fully1 = nn.Linear(num_inputs, num_hiddens)
        # self.bn1 = nn.BatchNorm1d(num_features=num_hiddens)
        self.Softmax = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):         
        ## x = F.relu(self.bn1(self.fully1(x))) # BatchNorm
        x = F.relu(self.fully1(x)) # no BatchNorm
        # print(x.shape) # shape: [16, 10] [batch, nh1]
        x = self.Softmax(x)
        return x

class mlp_2l(nn.Module):
    '''
    mlp: 2 hidden layers
    ref: https://github.com/Duncanswilson/maxout-pytorch/blob/master/maxout_pytorch.ipynb
    '''
    def __init__(self, num_inputs, 
                       nh1, 
                       nh2,
                       num_outputs):
        
        super(mlp_2l, self).__init__()
        self.fully1 = nn.Linear(num_inputs, nh1)
        self.fully2 = nn.Linear(nh1, nh2)
        self.Softmax = nn.Linear(nh2, num_outputs)

    def forward(self, x):         
        x = F.relu(self.fully1(x))
        x = F.relu(self.fully2(x))
        x = self.Softmax(x)
        return x

#%%
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.float().to(device)).argmax(dim=1) == y.long().to(device)).float().sum().cpu().item()
                net.train() 
            else: 
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def train_net(net, 
              train_iter, 
              test_iter, 
              loss, 
              num_epochs, 
              batch_size,
              params=None, 
              lr=None, 
              optimizer=None,
              usecuda=False,
              evaltest=True,
              num_class=20,
              binmean=None,
              pred=None,
              gt=None
              ):
    device = torch.device("cuda") if usecuda else torch.devise("cpu")    
    net.to(device)
    net.train() 
    ttart = time.time()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:            
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X) 
            l = loss(y_hat, y).sum() 
            
            assert optimizer is not None           
            # clear gradient
            optimizer.zero_grad()
            
            l.backward()
            optimizer.step()              
            
            train_l_sum += l.item()
                        
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item() 
            
            n += y.shape[0]
        # test_acc = evaluate_accuracy(test_iter, net) if evaltest else 0
        test_acc = evaluate_accuracy2(test_iter, net, num_class, usecuda, binmean, pred, gt)
        tend = time.time()
        print('epoch %d, loss %.5f, train acc %.3f, test acc %.3f, %.3f min'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, (tend-ttart)/60))

def evaluate_accuracy2(test_iter, net, num_class, usecuda, binmean, pred, gt):
    y_hat = evaluate_net(test_iter, net, num_class, usecuda)

    predmlp = binadd(y_hat, binmean, 20)
    pred['Yield_Mg_ha'] = predmlp
    prs = evaluatepoch(pred, gt)
    # print('pr: {:.4f}'.format(np.mean(prs)))
    return np.mean(prs)

def evaluate_accuracy3(test_iter, net, usecuda, pred, gt):
    y_hat = evaluate_net(test_iter, net, 1, usecuda)

    pred['Yield_Mg_ha'] = y_hat
    prs = evaluatepoch(pred, gt)
    # print('pr: {:.4f}'.format(np.mean(prs)))
    return np.mean(prs)

#%%
def train_reg(net, 
              train_iter, 
              test_iter, 
              loss, 
              num_epochs, 
              batch_size,
              params=None, 
              lr=None, 
              optimizer=None,
              usecuda=False,
              evaltest=True,
              num_class=20,
              binmean=None,
              pred=None,
              gt=None
              ):
    device = torch.device("cuda") if usecuda else torch.devise("cpu")    
    net.to(device)
    net.train() 
    ttart = time.time()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:            
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X) 
            l = loss(y_hat, y.unsqueeze(1)).sum() # regression
            
            assert optimizer is not None           
            # clear gradient
            optimizer.zero_grad()
            
            l.backward()
            optimizer.step()              
            
            train_l_sum += l.item()
                        
            train_acc_sum += pearsonr(y_hat.detach().cpu().numpy().flatten(), y.detach().cpu().numpy())[0] if len(y)>2 else 0 # regression
            
            n += 1
        # test_acc = evaluate_accuracy(test_iter, net) if evaltest else 0
        test_acc = evaluate_accuracy3(test_iter, net, usecuda, pred, gt) if evaltest else 0
        tend = time.time()
        print('epoch %d, loss %.5f, train acc %.3f, test acc %.3f, %.3f min'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, (tend-ttart)/60))
    return train_acc_sum / n
        
#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_net(data_iter, net, dim, usecuda=False):
    if usecuda:    
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.eval()
    y_hat = np.zeros((0, dim))
    with torch.no_grad():
        for X, y in data_iter:
            y_ = net(X.to(device))
            y_hat = np.vstack((y_hat, y_.detach().cpu().numpy()))
    return y_hat

def evaluate_mix(data_iter, net, dim, L, usecuda=False):
    if usecuda:    
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.eval()
    y_hats = [np.zeros((0, dim)) for l in range(L)]
    with torch.no_grad():
        for X, y, ID in data_iter:
            y_ = net(X.to(device))
            for l in range(L):
                y_hats[l] = np.vstack((y_hats[l], y_[l].detach().cpu().numpy()))
    return y_hats

def binadd(predmlp, binmean, topn):
    predmlp = softmax(predmlp, axis=1) 
    maxs = np.argsort(predmlp, axis=1)[:,-topn:]
    prds = np.sort(predmlp, axis=1)[:,-topn:]
    predmlp = np.sum(binmean[maxs]*prds, axis=1) / np.sum(prds, axis=1)
    return predmlp

#%% mixture net and training
class mlp_(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_branch_hiddens, num_outputs_list):
        """
        Args:
            num_inputs (int): Number of input features.
            num_hiddens (int): Number of neurons in the shared hidden layer.
            num_branch_hiddens (int): Number of neurons in each branch's hidden layer.
            num_outputs_list (list of int): List containing the number of output classes for each label.
        """
        super(mlp_, self).__init__()
        self.shared_layer = nn.Linear(num_inputs, num_hiddens)

        # Each branch has its own (Linear -> ReLU -> Linear) structure
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_hiddens, num_branch_hiddens),
                nn.ReLU(),
                nn.Linear(num_branch_hiddens, num_outputs)
            )
            for num_outputs in num_outputs_list
        ])

    def forward(self, x):
        x = F.relu(self.shared_layer(x))  # Shared layer
        outputs = [branch(x) for branch in self.branches]  # Branch-specific layers
        return outputs

#%%
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.fc = nn.Linear(input_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim))
            
    def forward(self, x):
        return self.fc(x)
        # return F.relu(self.fc(x))

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, num_hiddens):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(num_hiddens, output_dim) for _ in range(num_experts)])
        
        print('MoE net: input_dim {}, output_dim {}, num_experts {}, num_hiddens {}'.format(input_dim, output_dim, num_experts, num_hiddens))
        self.shared_layer = nn.Linear(input_dim, num_hiddens)

    def forward(self, x, idx):
        expert_input = F.relu(self.shared_layer(x))  # Shared layer
        
        expert_out = torch.stack([self.experts[idx[j]](expert_input[j].unsqueeze(0)) for j in range(x.shape[0])])
        
        expert_outputs = expert_out.squeeze(1)
        
        return expert_outputs

def train_moe(net, 
              train_iter, 
              test_iter, 
              loss, 
              num_epochs, 
              batch_size,
              params=None, 
              lr=None, 
              optimizer=None,
              usecuda=False,
              evaltest=True,
              num_class=20,
              pred=None,
              gt=None,
              reg=False
              ):
    device = torch.device("cuda") if usecuda else torch.device("cpu")
    net.to(device)
    net.train() 
    ttart = time.time()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y, ID in train_iter:                        
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X, ID) 
            l = loss(y_hat, y.unsqueeze(1)).sum() # regression
            
            assert optimizer is not None           
            # clear gradient
            optimizer.zero_grad()
            
            l.backward()
            optimizer.step()              
            
            train_l_sum += l.item()
                        
            train_acc_sum += pearsonr(y_hat.detach().cpu().numpy().flatten(), y.detach().cpu().numpy())[0] # regression
            
            n += 1 
        test_acc = evaluate_accuracy3(test_iter, net, usecuda, pred, gt) if evaltest else 0
        tend = time.time()
        print('epoch %d, loss %.5f, train acc %.3f, test acc %.3f, %.3f min'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, (tend-ttart)/60))

#%%
class MoEc(nn.Module):
    '''
    change: for one head data, also send to certain other heads
    add: id_send_to
    '''
    def __init__(self, input_dim, output_dim, num_experts, num_hiddens, 
                 id_send_to):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(num_hiddens, output_dim) for _ in range(num_experts)])
        
        print('MoE net: input_dim {}, output_dim {}, num_experts {}, num_hiddens {}'.format(input_dim, output_dim, num_experts, num_hiddens))
        self.shared_layer = nn.Linear(input_dim, num_hiddens)
        
        self.Wei = 1 # when doing output, use ensemble of multiple heads, but own head should be more weighted
        self.id_send_to = {}
        self.topk = max([len(v) for k,v in id_send_to.items()])
        self.weis = 1/self.topk
        for k,v in id_send_to.items():
            v = list(v)
            if len(v) == self.topk:
                self.id_send_to[k] = v
            else:
                v += [k]*(self.topk - len(v)) # is it 0, -1, or k?
                self.id_send_to[k] = v

    def forward(self, x, idx):
        ''' 
        idx: head id for this data
        '''
        expert_outputs = torch.zeros(x.shape[0], 1).to(x.device)
        x = F.relu(self.shared_layer(x))  # Shared layer        
        batch_idxs = [self.id_send_to[idx[j]] for j in range(x.shape[0])] # idx[j] is ID, get list        
        # print(batch_idxs)
        
        for i in range(self.topk):            
            expert_input = x.clone()
            _idx_ = [idxs[i] for idxs in batch_idxs]        
            expert_out = torch.stack([self.experts[_idx_[j]](expert_input[j].unsqueeze(0)) for j in range(x.shape[0])])
            # print(i, expert_out.shape, expert_out.squeeze(1))
            expert_outputs += expert_out.squeeze(1) * self.weis
        
        return expert_outputs

def train_moec(net, 
              train_iter, 
              test_iter, 
              loss, 
              num_epochs, 
              batch_size,
              params=None, 
              lr=None, 
              optimizer=None,
              usecuda=False,
              evaltest=True,
              num_class=20,
              pred=None,
              gt=None,
              reg=False
              ):
    device = torch.device("cuda") if usecuda else torch.device("cpu")
    net.to(device)
    net.train() 
    ttart = time.time()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y, ID in train_iter:                        
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X, ID.numpy()) # dataloader converted ID to tensor
            l = loss(y_hat, y.unsqueeze(1)).sum() # regression
            
            assert optimizer is not None           
            # clear gradient
            optimizer.zero_grad()
            
            l.backward()
            optimizer.step()              
            
            train_l_sum += l.item()
                        
            train_acc_sum += pearsonr(y_hat.detach().cpu().numpy().flatten(), y.detach().cpu().numpy())[0] # regression
            
            n += 1 
        test_acc = evaluate_accuracy3(test_iter, net, usecuda, pred, gt) if evaltest else 0
        tend = time.time()
        print('epoch %d, loss %.5f, train acc %.3f, test acc %.3f, %.3f min'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, (tend-ttart)/60))
#%%
def train_mix_(net, 
              train_iter, 
              test_iter, 
              loss, 
              num_epochs, 
              batch_size,
              params=None, 
              lr=None, 
              optimizer=None,
              usecuda=False,
              evaltest=True,
              num_class=20,
              binmean=None,
              pred=None,
              gt=None,
              reg=False
              ):
    '''
    actually, this doesn't really work
    because it just split batch predictions into 217 heads and do sum there, however, no difference w/ do loss sum all together
    '''
    device = torch.device("cuda") if usecuda else torch.device("cpu")
    net.to(device)
    net.train() 
    ttart = time.time()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y, ID in train_iter:                        
            X = X.to(device)
            y = y.to(device)                
            
            y_hats = net(X)  
            losses = []
            accs   = []
            for e in range(len(y_hats)):
                y_hat = y_hats[e]
                eid   = (ID == e)
                eid2  = (ID != e)
                if reg:
                    acc = pearsonr(y_hat.detach().cpu().numpy().flatten(), y.detach().cpu().numpy())[0]
                else:
                    acc = (y_hat.argmax(dim=1) == y)[eid].sum().item() # [eid]
                eid   = eid.to(torch.float32).to(device) # + eid2.to(torch.float32).to(device)*0.1
                if reg:
                    ll = loss(y_hat, y.unsqueeze(1))
                else:
                    ll = loss(y_hat, y)
                wed_l = (ll*eid).sum() # *eid
                # print(e, wed_l.detach().cpu().item())
                losses.append(wed_l) 
                accs.append(acc)
                # return 0
            l = sum(losses) / len(y_hats)
            
            # clear gradient
            assert optimizer is not None            
            optimizer.zero_grad()
            
            l.backward()            
            optimizer.step()                          
            train_l_sum += l.item()
                        
            train_acc_sum += sum(accs) / len(y_hats)
            
            n += 1 if reg else y.shape[0]
        if reg:
            test_acc = evaluate_accuracy3(test_iter, net, usecuda, pred, gt) if evaltest else 0
        else:
            # TODO
            test_acc = evaluate_accuracy2(test_iter, net, num_class, usecuda, binmean, pred, gt) if evaltest else 0
        tend = time.time()
        print('epoch %d, loss %.5f, train acc %.3f, test acc %.3f, %.3f min'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, (tend-ttart)/60))

#%%
