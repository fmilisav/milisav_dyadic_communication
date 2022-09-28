import numpy as np
from scipy.spatial.distance import *
import pandas as pd

from sklearn.model_selection import train_test_split
from nilearn.surface import load_surf_data

import bct
from netneurotools import networks as nets
from netneurotools import metrics, datasets
from strength_preserving_rand import strength_preserving_rand
from match_length_degree_distribution import match_length_degree_distribution
from path_transitivity import path_transitivity

import os
import copy
import sys
sys.path.append('..')
from utils import pickle_dump
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

cortical125 = np.loadtxt(os.path.abspath('../../data/original_data/Lausanne/'
                                         'cortical/cortical125.txt'))
cortical500 = np.loadtxt(os.path.abspath('../../data/original_data/Lausanne/'
                                         'cortical/cortical500.txt'))
cor_idx125 = [i for i, val in enumerate(cortical125) if val == 1]
cor_idx500 = [i for i, val in enumerate(cortical500) if val == 1]

#upper triangle masks
mask125 = np.triu(np.ones((len(cor_idx125), len(cor_idx125))), k=1) > 0
mask500 = np.triu(np.ones((len(cor_idx500), len(cor_idx500))), k=1) > 0

coords125 = np.load('../../data/original_data/Lausanne/coords/coords125.npy')
coords500 = np.load('../../data/original_data/Lausanne/coords/coords500.npy')
cor_coords125 = coords125[cor_idx125]
cor_coords500 = coords500[cor_idx500]

#Euclidean distance
euc_dist125 = squareform(pdist(cor_coords125))
euc_dist500 = squareform(pdist(cor_coords500))

struct_den_scale125 = np.load('../../data/original_data/Lausanne/struct/'
                              'struct_den_scale125.npy')
struct_den_scale500 = np.load('../../data/original_data/Lausanne/struct/'
                              'struct_den_scale500.npy')
cor_struct_den_scale125 = struct_den_scale125[cor_idx125][:, cor_idx125]
#Taking out participant with missing functional data
cor_struct_den_scale125 = np.delete(cor_struct_den_scale125, 33, axis = 2)
cor_struct_den_scale500 = struct_den_scale500[cor_idx500][:, cor_idx500]
#Taking out participant with missing functional data
cor_struct_den_scale500 = np.delete(cor_struct_den_scale500, 33, axis = 2)

n = cor_struct_den_scale125.shape[2]

del struct_den_scale125
del struct_den_scale500

#preprocessed functional time-series
func_scale125 = np.load('../../data/original_data/Lausanne/time_series/'
                        'func_scale125.npy')
func_scale500 = np.load('../../data/original_data/Lausanne/time_series/'
                        'func_scale500.npy')
cor_func_scale125 = func_scale125[cor_idx125]
#Taking out participant with missing functional data
cor_func_scale125 = np.delete(cor_func_scale125, 33, axis = 2)
cor_func_scale500 = func_scale500[cor_idx500]
#Taking out participant with missing functional data
cor_func_scale500 = np.delete(cor_func_scale500, 33, axis = 2)

del func_scale125
del func_scale500

struct_den_dict = {'125': cor_struct_den_scale125,
                   '500': cor_struct_den_scale500}
func_dict = {'125': cor_func_scale125, '500': cor_func_scale500}

del cor_struct_den_scale125
del cor_struct_den_scale500
del cor_func_scale125
del cor_func_scale500

first_right_sub_id125 = np.argwhere(cortical125 == 0)[0, 0]
len_left125 = len(cor_idx125) - first_right_sub_id125
hemiid125 = np.array(first_right_sub_id125*[0] + len_left125*[1])
hemiid125 = hemiid125[:, np.newaxis] #hemispheric labels for struct_consensus

first_right_sub_id500 = np.argwhere(cortical500 == 0)[0, 0]
len_left500 = len(cor_idx500) - first_right_sub_id500
hemiid500 = np.array(first_right_sub_id500*[0] + len_left500*[1])
hemiid500 = hemiid500[:, np.newaxis] #hemispheric labels for struct_consensus


#Splitting the sample into Discovery and Validation subsets
random_state = 0
discov_valid_split = train_test_split(range(n), test_size = 0.5,
                                      random_state = random_state)

temp_struct_den_dict = {}
for key, value in struct_den_dict.items():
    temp_struct_den_dict[key + '_discov'] = value[:, :, discov_valid_split[0]]
    temp_struct_den_dict[key + '_valid'] = value[:, :, discov_valid_split[1]]

temp_func_dict = {}
for key, value in func_dict.items():
    temp_func_dict[key + '_discov'] = value[:, :, discov_valid_split[0]]
    temp_func_dict[key + '_valid'] = value[:, :, discov_valid_split[1]]


#Generating consensus networks
#-------------------------------------------------------------------------------
#Generating structural consensus
temp_dict = {}
for key, value in temp_struct_den_dict.items():
    mean = np.mean(value, axis = 2)
    #lower resolution
    if '125' in key:
        #binary
        temp_dict[key + '_bin'] = nets.struct_consensus(value, euc_dist125,
                                                        hemiid125)
        #check connectedness
        num_comp = bct.number_of_components(temp_dict[key + '_bin'])
        assert num_comp == 1,'{}: disconnected'.format(key)
        #weighted
        temp_dict[key + '_wei'] = temp_dict[key + '_bin'] * mean
    #higher resolution
    else:
        #binary
        temp_dict[key + '_bin'] = nets.struct_consensus(value, euc_dist500,
                                                        hemiid500)
        #check connectedness
        num_comp = bct.number_of_components(temp_dict[key + '_bin'])
        assert num_comp == 1, '{}: disconnected'.format(key)
        #weighted
        temp_dict[key + '_wei'] = temp_dict[key + '_bin'] * mean

sensitivity_keys = ['125_discov_wei', '500_discov_bin',
                    '500_discov_wei', '500_valid_wei']

#Removing data splits that won't be used in the analysis
keys = list(temp_dict.keys())
for key in keys:
    if key not in sensitivity_keys:
        temp_dict.pop(key)
struct_den_dict = temp_dict
pickle_dump('struct_den_dict', struct_den_dict)

#Generating functional consensus
temp_dict = {}
for key, value in temp_func_dict.items():
    #subject-wise functional connectivity matrices
    corrs = [np.corrcoef(value[..., sub]) for sub in range(value.shape[-1])]
    #Fisher's r-to-z transformation
    corrs = np.arctanh(corrs)
    #functional consensus
    mean = np.nanmean(corrs, axis = 0)
    #z-to-r back-transformation
    consensus = np.tanh(mean)
    temp_dict[key] = consensus

sensitivity_keys = ['500_discov']

#Removing data splits that won't be used in the analysis
keys = list(temp_dict.keys())
for key in keys:
    if key not in sensitivity_keys:
        temp_dict.pop(key)
func_dict = temp_dict
pickle_dump('func_dict', func_dict)

del func_dict


#Computing topological features
#-------------------------------------------------------------------------------
#strength
strengths_dict = {}
for key, value in struct_den_dict.items():
    if 'wei' not in key: continue
    strengths_dict[key] = bct.strengths_und(value)

#betweenness centrality
betweenness_dict = {}
for key, value in struct_den_dict.items():
    if 'bin' in key:
        betweenness_dict[key] = np.log(bct.betweenness_bin(value))
    else: betweenness_dict[key] = np.log(bct.betweenness_wei(-np.log(value)))

#netneurotools Cammoun info
cammoun_info = pd.read_csv(datasets.fetch_cammoun2012()['info'])
df = cammoun_info.copy()

#participation coefficient
participation_dict = {}
for key, value in struct_den_dict.items():
    scale = 'scale125' if '125' in key else 'scale500'
    df_rows_bool_arr = (df['structure'] == 'cortex') & (df['scale'] == scale)
    yeo_7 = df.loc[df_rows_bool_arr]['yeo_7'].values
    participation_dict[key] = bct.participation_coef(value, yeo_7)


#Generating nulls
#-------------------------------------------------------------------------------
#Generating randomized nulls
rand_struct_den_dict = {}
nrand = 100 #number of nulls
niter = 10 #each edge is rewired approximately 10 times
for key, value in struct_den_dict.items():
    if 'wei' in key:
        #strength sequence-preserving randomization for weighted networks
        randmio_func = strength_preserving_rand
    else:
        #Maslov & Sneppen degree-preserving rewiring for binary networks
        randmio_func = bct.randmio_und_connected
    #process-based parallelization
    rand_struct_den = Parallel(n_jobs = 12, verbose = 1)(delayed(randmio_func)(value, niter, seed = i) for i in range(nrand))
    rand_struct_den, _ = zip(*rand_struct_den)
    rand_struct_den = np.dstack(rand_struct_den)
    rand_struct_den_dict[key] = rand_struct_den
    print(key + ': randomization done')

rand_struct_den = rand_struct_den_dict['500_discov_wei'][:, :, :3]
pickle_dump('rand_struct_den', rand_struct_den)

#Generating geometry-preserving randomized nulls
geo_rand_struct_den_dict = {}
nrand = 100 #number of nulls
key = '500_discov_wei'
value = struct_den_dict[key]
D = euc_dist125 if '125' in key else euc_dist500
k = int(np.count_nonzero(value)/2)
nbins = int(np.sqrt(k))
nswap = len(value)*20

geo_rand_struct_den = Parallel(n_jobs = 12, verbose = 1)(delayed(match_length_degree_distribution)(value, D,
                                                                                                 nbins = nbins,
                                                                                                 nswap = nswap,
                                                                                                 weighted = True,
                                                                                                 seed = i) for i in range(nrand))
geo_rand_struct_den_bin, geo_rand_struct_den_wei, _ = zip(*geo_rand_struct_den)
geo_rand_struct_den = np.dstack(geo_rand_struct_den_wei)
geo_rand_struct_den_dict[key] = geo_rand_struct_den


#Computing communication distances
#-------------------------------------------------------------------------------
comm_mods_dicts = {'shortest paths': {}, 'search information': {},
                   'path transitivity': {}, 'communicability': {},
                   'mean first-passage time': {}}

comm_mods_fcts = {'shortest paths': [bct.distance_bin, bct.distance_wei],
                  'search information': bct.search_information,
                  'path transitivity': path_transitivity,
                  'communicability': [metrics.communicability_bin,
                                      metrics.communicability_wei],
                  'mean first-passage time': bct.mean_first_passage_time}

#paralellization wrapper
@delayed
@wrap_non_picklable_objects
def comm_mods_dists_wrap(key_comm_mod, fct, key, value):
    return comm_mods_dists(key_comm_mod, fct, key, value)

#function for computing communication distance based on
#a given communication model
def comm_mods_dists(key_comm_mod, fct, key, value):
    #check if the key corresponds to the considered models
    if (key_comm_mod != 'shortest paths' and
        key_comm_mod != 'search information'
        and key_comm_mod != 'path transitivity' and
        key_comm_mod != 'communicability' and
        key_comm_mod != 'mean first-passage time'):
        raise ValueError('unknown communication model')
    #for binary networks
    if 'bin' in key:
        if (key_comm_mod == 'shortest paths' or
            key_comm_mod == 'communicability'): dist = fct[0](value)
        else: dist = fct(value)
    #for weighted networks
    else:
        if key_comm_mod == 'shortest paths': dist = fct[1](-np.log(value))[0]
        elif key_comm_mod == 'communicability': dist = fct[1](value)
        elif (key_comm_mod == 'path transitivity' or
              key_comm_mod == 'search information'): dist = fct(value, 'log')
        else: dist = fct(value)
    return dist

def comm_mods_dist_dicts(struct_den_dict, type,
                         comm_mods_dist_dicts,
                         comm_mods_fcts = comm_mods_fcts,
                         nrand = 100):
    for key_comm_mod in comm_mods_dist_dicts.keys():
        #communication function
        fct = comm_mods_fcts[key_comm_mod]
        #data subset
        for key, value in struct_den_dict.items():
            if type == 'original':
                dist = comm_mods_dists(key_comm_mod, fct, key, value)
            #paralellization for null networks
            elif type == 'rand' or type == 'geo_rand':
                dist = Parallel(n_jobs = 12, verbose = 1)(comm_mods_dists_wrap(key_comm_mod, fct, key, value[:, :, i]) for i in range(nrand))
                dist = np.dstack(dist)
            else: raise ValueError('unknown type')
            comm_mods_dist_dicts[key_comm_mod][key] = dist
            print(key + ': done')
        print(key_comm_mod + ': done')

#empirical communication distance
dist_dicts = copy.deepcopy(comm_mods_dicts)
comm_mods_dist_dicts(struct_den_dict, 'original', dist_dicts)
pickle_dump('dist_dicts', dist_dicts)
del struct_den_dict

#null communication distances
rand_dist_dicts = copy.deepcopy(comm_mods_dicts)
comm_mods_dist_dicts(rand_struct_den_dict, 'rand', rand_dist_dicts)
rand_dist = rand_dist_dicts['shortest paths']['500_discov_wei'][:, :, :3]
pickle_dump('rand_dist', rand_dist)
del rand_struct_den_dict

#mean null communication distance
mean_dist_dicts = copy.deepcopy(comm_mods_dicts)
for key_comm_mod in comm_mods_dicts.keys():
    for key, value in rand_dist_dicts[key_comm_mod].items():
        mean_dist_dicts[key_comm_mod][key] = np.mean(value, axis = 2)
pickle_dump('mean_dist_dicts', mean_dist_dicts)

#standard deviation of null communication distances
std_dist_dicts = copy.deepcopy(comm_mods_dicts)
for key_comm_mod in comm_mods_dicts.keys():
    for key, value in rand_dist_dicts[key_comm_mod].items():
        std_dist_dicts[key_comm_mod][key] = np.std(value, axis = 2, ddof = 1)
pickle_dump('std_dist_dicts', std_dist_dicts)

del rand_dist_dicts

#geometry-preserving null communication distances
geo_rand_dist_dicts = copy.deepcopy(comm_mods_dicts)
comm_mods_dist_dicts(geo_rand_struct_den_dict, 'geo_rand', geo_rand_dist_dicts)
del geo_rand_struct_den_dict

#mean geometry-preserving null communication distances
geo_mean_dist_dicts = copy.deepcopy(comm_mods_dicts)
for key_comm_mod in comm_mods_dicts.keys():
    for key, value in geo_rand_dist_dicts[key_comm_mod].items():
        geo_mean_dist_dicts[key_comm_mod][key] = np.mean(value, axis = 2)
pickle_dump('geo_mean_dist_dicts', geo_mean_dist_dicts)

#standard deviation of geometry-preserving null communication distances
geo_std_dist_dicts = copy.deepcopy(comm_mods_dicts)
for key_comm_mod in comm_mods_dicts.keys():
    for key, value in geo_rand_dist_dicts[key_comm_mod].items():
        geo_std_dist_dicts[key_comm_mod][key] = np.std(value, axis = 2, ddof = 1)
pickle_dump('geo_std_dist_dicts', geo_std_dist_dicts)

del geo_rand_dist_dicts

#null-normalized communication distances
norm_dist_dicts = copy.deepcopy(comm_mods_dicts)
for key_comm_mod in comm_mods_dicts.keys():
    for key, value in dist_dicts[key_comm_mod].items():
        demeaned = value - mean_dist_dicts[key_comm_mod][key]
        norm = demeaned/std_dist_dicts[key_comm_mod][key]
        norm_dist_dicts[key_comm_mod][key] = norm

del mean_dist_dicts
pickle_dump('norm_dist_dicts', norm_dist_dicts)

#geometry-preserving null-normalized communication distances
geo_norm_dist_dicts = copy.deepcopy(comm_mods_dicts)
for key_comm_mod in comm_mods_dicts.keys():
    key = '500_discov_wei'
    value = dist_dicts[key_comm_mod][key]
    demeaned = value - geo_mean_dist_dicts[key_comm_mod][key]
    norm = demeaned/std_dist_dicts[key_comm_mod][key]
    geo_norm_dist_dicts[key_comm_mod][key] = norm

del geo_mean_dist_dicts
pickle_dump('geo_norm_dist_dicts', geo_norm_dist_dicts)

del dist_dicts

#node-mean null-normalized communication distances
node_mean_norm_dist_dicts = copy.deepcopy(comm_mods_dicts)
for key_comm_mod in comm_mods_dicts.keys():
    for key, value in norm_dist_dicts[key_comm_mod].items():
        mean = np.nanmean(value, axis = 1)
        node_mean_norm_dist_dicts[key_comm_mod][key] = mean

del norm_dist_dicts

pickle_dump('node_mean_norm_dist_dicts', node_mean_norm_dist_dicts)
del node_mean_norm_dist_dicts

#node-mean geometry-preserving null-normalized communication distances
node_mean_geo_norm_dist_dicts = copy.deepcopy(comm_mods_dicts)
for key_comm_mod in comm_mods_dicts.keys():
    for key, value in geo_norm_dist_dicts[key_comm_mod].items():
        mean = np.nanmean(value, axis = 1)
        node_mean_geo_norm_dist_dicts[key_comm_mod][key] = mean

del geo_norm_dist_dicts

pickle_dump('node_mean_geo_norm_dist_dicts', node_mean_geo_norm_dist_dicts)
del node_mean_geo_norm_dist_dicts


#Mapping Cammoun parcels from the Neurosynth maps to the netneurotools info
#-------------------------------------------------------------------------------
cammoun_ns_dir = '../../data/original_data/cammoun_ns/'

#Neurosynth maps dataframes
cammoun_ns_125_df = pd.read_csv(cammoun_ns_dir + 'scale125.csv', index_col = 0)
cammoun_ns_500_df = pd.read_csv(cammoun_ns_dir + 'scale500.csv', index_col = 0)

df = cammoun_info.copy()
#indices for mapping Cammoun parcels across dataframes
cammoun_ns_df_idx = {'scale125': [], 'scale500': []}
for scale in cammoun_ns_df_idx.keys():
    df_rows_bool_arr = (df['structure'] == 'cortex') & (df['scale'] == scale)
    cammoun_ns_df=cammoun_ns_125_df if scale=='scale125' else cammoun_ns_500_df
    #across regions from the first dataframe
    for region in df.loc[df_rows_bool_arr].index:
        label = df.loc[region].label
        hemi = df.loc[region].hemisphere
        hemi_id = 'rh' if hemi == 'R' else 'lh'
        #finding the same region from the same hemisphere in the other dataframe
        for idx in cammoun_ns_df.index:
            if hemi_id in idx and label in idx:
                cammoun_ns_df_idx[scale].append(idx)
                break


#Dumping preprocessed data
#-------------------------------------------------------------------------------
objs_to_dump = {'mask125': mask125, 'mask500': mask500,
                'euc_dist125': euc_dist125, 'euc_dist500': euc_dist500,
                'strengths_dict': strengths_dict,
                'betweenness_dict': betweenness_dict,
                'participation_dict': participation_dict,
                'cammoun_ns_df_idx': cammoun_ns_df_idx}

for key, value in objs_to_dump.items():
    pickle_dump(key, value)
