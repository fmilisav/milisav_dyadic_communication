import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd
from scipy.stats import zscore, spearmanr, rankdata
from sklearn import preprocessing

import seaborn as sns
sns.set_style("ticks")
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.colors as mcolors
from netneurotools import datasets, plotting, freesurfer, stats
from mayavi import mlab
mlab.options.offscreen = True

from nilearn.plotting import plot_connectome

import os
import sys
sys.path.append('..')
import copy
from utils import pickle_load
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('pickle')

from analysis_utils import *

mask125 = pickle_load('mask125')
mask500 = pickle_load('mask500')

euc_dist125 = pickle_load('euc_dist125')
euc_dist500 = pickle_load('euc_dist500')

struct_den_dict = pickle_load('struct_den_dict')
rand_struct_den = pickle_load('rand_struct_den')
func_dict = pickle_load('func_dict')

strengths_dict = pickle_load('strengths_dict')
betweenness_dict = pickle_load('betweenness_dict')
participation_dict = pickle_load('participation_dict')

cammoun_ns_df_idx = pickle_load('cammoun_ns_df_idx')

dist_dicts = pickle_load('dist_dicts')
rand_dist = pickle_load('rand_dist')
mean_dist_dicts = pickle_load('mean_dist_dicts')
geo_mean_dist_dicts = pickle_load('geo_mean_dist_dicts')
norm_dist_dicts = pickle_load('norm_dist_dicts')
geo_norm_dist_dicts = pickle_load('geo_norm_dist_dicts')
node_mean_norm_dist_dicts = pickle_load('node_mean_norm_dist_dicts')
node_mean_geo_norm_dist_dicts = pickle_load('node_mean_geo_norm_dist_dicts')

comm_mod = 'shortest paths'
key = '500_discov_wei'

norm_dists = {'rand': norm_dist_dicts[comm_mod][key],
              'geo_rand': geo_norm_dist_dicts[comm_mod][key]}

mono_idx = struct_den_dict[key].copy()
#Mapping monosynaptic dyads to 1 and polysynaptic dyads to 0
mono_idx[mono_idx > 0] = 1
#Creating mask
mono_idx = np.nonzero(mono_idx)

poly_node_mean_dicts = {'rand': {comm_mod: {}},
                        'geo_rand': {comm_mod: {}}}
for null_mod, value in norm_dists.items():

    value = value.copy()
    #Taking out monosynaptic dyads
    value[mono_idx] = np.nan
    poly_node_mean_dicts[null_mod][comm_mod][key] = np.nanmean(value, axis = 1)

#unexpectedly short path lengths
neg_norm_dist = norm_dist_dicts[comm_mod][key].copy()
neg_norm_dist[neg_norm_dist > 0] = np.nan
poly_neg_norm_dist = neg_norm_dist.copy()
poly_neg_norm_dist[mono_idx] = np.nan

#cortical regions
cortical500 = np.loadtxt(os.path.abspath('../../data/original_data/Lausanne/'
                                         'cortical/cortical500.txt'))
cor_idx500 = [i for i, val in enumerate(cortical500) if val == 1]

#regional coordinates
coords500 = np.load('../../data/original_data/Lausanne/coords/coords500.npy')

#cortical coordinates
cor_coords500 = coords500[cor_idx500]

#figures output directory
dir = os.path.abspath('../../figures')

pca_norm_dist_dicts = copy.deepcopy(norm_dist_dicts)
mfpt_norm_dist_dicts = mfpt_zscore(pca_norm_dist_dicts)
z_norm_dist_dicts = zscore_dist_dicts(mfpt_norm_dist_dicts, 'edge')
pca_scores_dict = pca_dist_dicts(z_norm_dist_dicts, 'edge', dir)

symmetrise(node_mean_norm_dist_dicts, mfpt_norm_dist_dicts)
pca_node_mean_norm_dist_dicts = copy.deepcopy(node_mean_norm_dist_dicts)
z_node_mean_norm_dist_dicts = zscore_dist_dicts(pca_node_mean_norm_dist_dicts,
                                                'node')
pca_nodal_scores_dict = pca_dist_dicts(z_node_mean_norm_dist_dicts, 'node', dir)

norm_dist_dicts['pc1'] = pca_scores_dict['pc1']
node_mean_norm_dist_dicts['pc1'] = pca_nodal_scores_dict['pc1']

#communication models considered
comm_mods = ['shortest paths', 'pc1']

#shorter-than-expected dyads
num = sum(norm_dist_dicts['shortest paths']['500_discov_wei'].ravel() < 0)
#total - undefined diagonal elements
denom = len(norm_dist_dicts['shortest paths']['500_discov_wei'].ravel()) - 1000
print(('shorter than expected percentage: '
      '{}%').format(num/denom*100))
#shorter-than-expected dyads
num = sum(geo_norm_dist_dicts['shortest paths']['500_discov_wei'].ravel() < 0)
#total - undefined diagonal elements
denom=len(geo_norm_dist_dicts['shortest paths']['500_discov_wei'].ravel()) -1000
print(('shorter than expected percentage (geometry-preserving): '
      '{}%').format(num/denom*100))

#useful data for partition specificity analyses among Yeo networks
partition_yeo_7_dict = {'idx': {}, 'yeo_7': {}, 'rsn_idx': {},
                        'reorder_idx': {}, 'w_b_yeo': {}}
for res in (125, 500):
    nodes = 1000 if res == 500 else 219
    idx, yeo_7, rsn_idx, reorder_idx = partition_yeo_7(res)
    partition_yeo_7_dict['idx'][res] = idx
    partition_yeo_7_dict['yeo_7'][res] = yeo_7
    partition_yeo_7_dict['rsn_idx'][res] = rsn_idx
    partition_yeo_7_dict['reorder_idx'][res] = reorder_idx

#Matrices
#-------------------------------------------------------------------------------
matrices_path = os.path.join(dir, 'matrices')
make_dir(matrices_path)
matrices = {'structural_density': struct_den_dict['500_discov_wei'],
            'rewired_structural_density_1': rand_struct_den[:, :, 0],
            'rewired_structural_density_2': rand_struct_den[:, :, 1],
            'rewired_structural_density_3': rand_struct_den[:, :, 2],
            'original_shortest_path_lengths':
            dist_dicts['shortest paths']['500_discov_wei'],
            'rewired_shortest_path_lengths_1': rand_dist[:, :, 0],
            'rewired_shortest_path_lengths_2': rand_dist[:, :, 1],
            'rewired_shortest_path_lengths_3': rand_dist[:, :, 2],
            'standardized_shortest_path_lengths':
            norm_dist_dicts['shortest paths']['500_discov_wei'],
            'standardized_sp':
            z_norm_dist_dicts['shortest paths']['500_discov_wei'],
            'standardized_si':
            z_norm_dist_dicts['search information']['500_discov_wei'],
            'standardized_pt':
            z_norm_dist_dicts['path transitivity']['500_discov_wei'],
            'standardized_com':
            z_norm_dist_dicts['communicability']['500_discov_wei'],
            'standardized_mfpt':
            z_norm_dist_dicts['mean first-passage time']['500_discov_wei']}

for key, value in matrices.items():

    matrix_plot_path = os.path.join(matrices_path, key + '.svg')

    if (key == 'structural_density' or
        key == 'rewired_structural_density_1' or
        key == 'rewired_structural_density_2' or
        key == 'rewired_structural_density_3'):
        cmap = 'OrRd'
        vmin = 0
        vmax = 0.0001
    elif (key == 'original_shortest_path_lengths' or
          key == 'rewired_shortest_path_lengths_1' or
          key == 'rewired_shortest_path_lengths_2' or
          key == 'rewired_shortest_path_lengths_3'):
        cmap = 'OrRd_r'
        vmin = 0
        vmax = 25
    elif key == 'standardized_shortest_path_lengths':
        cmap = 'RdBu_r'
    else:
        cmap = 'RdBu_r'
        vmin = -3
        vmax = 3

    fig, ax = plt.subplots(1, 1)
    if key == 'standardized_shortest_path_lengths':
        coll = ax.imshow(value, cmap = cmap,
                         norm = mcolors.TwoSlopeNorm(0, vmin = -5, vmax = 10))
    else: coll = ax.imshow(value, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.axis('off')
    ax.set(xticklabels=[], yticklabels=[])
    ax.set(xticks=[], yticks=[])
    cb = fig.colorbar(coll, orientation = 'horizontal')
    cb.set_ticks([])
    cb.outline.set_visible(False)
    fig.savefig(matrix_plot_path, dpi = 300)
    plt.close(fig)

#modular matrices
nets_reorder_idx = [6, 4, 1, 5, 3, 2, 0] #reordering index for Yeo networks
dyad_label = 'standardized shortest path length'
for comm_mod in comm_mods:

    mod_matrix_plot_path= os.path.join(matrices_path,comm_mod+'_mod_matrix.png')

    idx = partition_yeo_7_dict['idx'][500]
    reorder_idx = partition_yeo_7_dict['reorder_idx'][500]
    yeo_7 = partition_yeo_7_dict['yeo_7'][500]
    le = preprocessing.LabelEncoder()
    #community labels
    yeo_7_encoded = le.fit_transform(yeo_7)
    yeo_7_encoded_reordered=np.array(list(map(lambda idx: nets_reorder_idx[idx],
                                              yeo_7_encoded)))
    yeo_7_encoded_reordered += 1


    if comm_mod == 'shortest paths':
        norm = mcolors.TwoSlopeNorm(0, vmin = -5, vmax = 10)
    else:
        norm = mcolors.TwoSlopeNorm(0, vmin = -3, vmax = 3)

    value = norm_dist_dicts[comm_mod]['500_discov_wei'].copy()
    #Reordering by Yeo network affiliation
    reordered_val = value[idx][:, idx][reorder_idx][:, reorder_idx]
    ylabels = ['vis', 'sm', 'da', 'va', 'lim', 'fp', 'dm']
    mappable = plotting.plot_mod_heatmap(reordered_val,
                                         yeo_7_encoded_reordered,
                                         inds = range(len(value)),
                                         ylabels = ylabels, cbar = False,
                                         norm = norm, cmap = 'RdBu_r')
    mappable.set_xticks([])
    mappable.set_xticklabels([])

    if comm_mod == 'shortest paths': label = 'standardized shortest path length'
    elif comm_mod == 'pc1': label = 'PC1 score'
    cb = plt.colorbar(cm.ScalarMappable(norm = norm, cmap = 'RdBu_r'),
                      ax = mappable, label = dyad_label,
                      orientation = 'horizontal')
    cb.set_ticks([])
    cb.outline.set_visible(False)

    mappable.set_ylabel('intrinsic networks')
    fig = mappable.get_figure()
    fig.savefig(mod_matrix_plot_path, bbox_inches='tight', dpi = 300)
    plt.close(fig)
plt.close('all')

#Bivariate histograms
#-------------------------------------------------------------------------------
bivar_hists_path = os.path.join(dir, 'bivariate histograms')
make_dir(bivar_hists_path)

y_dicts = {'rand': mean_dist_dicts, 'geo_rand': geo_mean_dist_dicts}
for null_mod, data_dict in y_dicts.items():
    bivar_hists_null_mod_path = os.path.join(bivar_hists_path, null_mod)
    make_dir(bivar_hists_null_mod_path)
    for key, value in data_dict['shortest paths'].items():
        if null_mod == 'rand' and key == '500_discov_bin':
            continue
        bivar_hists_split_path = os.path.join(bivar_hists_null_mod_path,
                                              key + '.png')

        mask = mask125 if '125' in key else mask500
        x, y = masking(dist_dicts['shortest paths'][key], 'shortest paths',
                       mask = mask, y = value)

        g = bivar_hists(x, y)

        xlim = g.ax_joint.get_xlim()
        ylim = g.ax_joint.get_ylim()
        lims = [np.min([xlim, ylim]),
                np.max([xlim, ylim])]

        #plotting identity line
        g.ax_joint.plot(lims, lims, 'k-', zorder = 0)

        set_joint_plot_lims(g, lims)
        set_joint_plot_pos(g)

        g.savefig(bivar_hists_split_path, dpi = 300)
        plt.close(g.fig)

        if null_mod == 'rand' and key == '500_discov_wei':
            #Coloring based on the partition between
            #monosynaptic and polysynaptic dyads
            bivar_hists_split_path = os.path.join(bivar_hists_null_mod_path,
                                                  key + '_partition_conn.png')
            hue = struct_den_dict[key].copy()
            #Mapping monosynaptic dyads to 1 and polysynaptic dyads to 0
            hue[hue > 0] = 1
            hue, _ = masking(hue, 'shortest paths', mask = mask)
            g = bivar_hists(x, y, hue = hue, hue_order = [1, 0],
                            palette = ['orange', 'royalblue'])
            xlim = g.ax_joint.get_xlim()
            ylim = g.ax_joint.get_ylim()
            lims = [np.min([xlim, ylim]),
                    np.max([xlim, ylim])]
            g.ax_joint.plot(lims, lims, 'k-', zorder = 0)
            set_joint_plot_lims(g, lims)
            set_joint_plot_pos(g)
            legend = g.ax_joint.get_legend()
            handles = legend.legendHandles
            legend.remove()
            g.ax_joint.legend(handles, ['monosynaptic', 'polysynaptic'],
                              bbox_to_anchor = (0, 0.8, 1, 0.1),
                              loc = 'lower left')
            g.savefig(bivar_hists_split_path, dpi = 300)
            plt.close(g.fig)

#Histograms
#-------------------------------------------------------------------------------
hists_path = os.path.join(dir, 'histograms')
make_dir(hists_path)

for key in ['500_discov_bin', '500_discov_wei']:

    value = norm_dist_dicts['shortest paths'][key].copy()

    hists_split_path = os.path.join(hists_path, key + '.svg')

    mask = mask125 if '125' in key else mask500
    x, _ = masking(value, 'shortest paths', mask = mask)

    ax = sns.histplot(x = x, bins = 'sqrt', element = 'step')
    ax.set_xlabel('standardized shortest path length')
    ax.set_box_aspect(1)
    sns.despine(ax = ax)
    save_plot(ax, hists_split_path)

    if key == '500_discov_wei':
        #Coloring based on the partition between
        #monosynaptic and polysynaptic dyads
        hists_split_path = os.path.join(hists_path, key + '_partition_conn.svg')
        hue = struct_den_dict[key].copy()
        #Mapping monosynaptic dyads to 1 and polysynaptic dyads to 0
        hue[hue > 0] = 1
        x, y = masking(value, 'shortest paths', mask = mask, y = hue)
        ax = sns.histplot(x = x, bins = 'sqrt', hue = y, hue_order = [1, 0],
                          element = 'step', palette = ['orange', 'royalblue'])
        legend = ax.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax.legend(handles, ['monosynaptic', 'polysynaptic'],
                  bbox_to_anchor = (0, 1.01, 1, 0.1), loc = 'lower left')
        ax.set_xlabel('standardized shortest path length')
        ax.set_box_aspect(1)
        sns.despine(ax = ax)
        save_plot(ax, hists_split_path)

#.annot files in fsaverage space for Cammoun parcellations
cammoun125 = datasets.fetch_cammoun2012('fsaverage')['scale125']
cammoun500 = datasets.fetch_cammoun2012('fsaverage')['scale500']

#Brain maps
#-------------------------------------------------------------------------------
brain_maps_path = os.path.join(dir, 'brain maps')
make_dir(brain_maps_path)

node_mean_maps_path = os.path.join(brain_maps_path, 'node_mean_std_spl')
make_dir(node_mean_maps_path)
pca_maps_path = os.path.join(brain_maps_path, 'pca_maps')
make_dir(pca_maps_path)

#plotting unexpectedly short paths
fig, ax = plt.subplots(1, figsize = (10, 10))
ax.axis("off")
output_file = str(os.path.join(node_mean_maps_path,
                               'rand_std_spl_500_discov_wei_all.png'))
plot_connectome(-1*neg_norm_dist, cor_coords500, node_color = 'black',
                node_size = 1, edge_cmap = 'OrRd', edge_threshold = '99%',
                annotate = False, alpha = 0.7,
                edge_vmin = np.nanpercentile(-1*neg_norm_dist, 99),
                edge_vmax = np.nanmax(-1*neg_norm_dist),
                figure = fig, output_file = output_file)
plt.close(fig)

fig, ax = plt.subplots(1, figsize = (10, 10))
ax.axis("off")
output_file = str(os.path.join(node_mean_maps_path,
                               'rand_std_spl_500_discov_wei_poly.png'))
plot_connectome(-1*poly_neg_norm_dist, cor_coords500, node_color = 'black',
                node_size = 1, edge_cmap = 'rainbow', edge_threshold = '99%',
                annotate = False, alpha = 0.7,
                edge_vmin = np.nanpercentile(-1*poly_neg_norm_dist, 99),
                edge_vmax = np.nanmax(-1*poly_neg_norm_dist),
                figure = fig, output_file = output_file)
plt.close(fig)


brain_maps = {'pc1_score': node_mean_norm_dist_dicts['pc1']['500_discov_wei'],
              'node_mean_geo_rand_std_spl_500_discov_wei':
              node_mean_geo_norm_dist_dicts['shortest paths']['500_discov_wei'],
              'node_mean_rand_std_spl_500_discov_wei_poly':
              poly_node_mean_dicts['rand']['shortest paths']['500_discov_wei'],
              'node_mean_geo_rand_std_spl_500_discov_wei_poly':
              poly_node_mean_dicts['geo_rand']['shortest paths']['500_discov_wei']}

for key, value in node_mean_norm_dist_dicts['shortest paths'].items():
    brain_maps['node_mean_standardized_shortest_path_lengths_' + key] = value

comm_mod_labels = ['sp', 'si', 'pt', 'com', 'mfpt']
for comm_mod in range(len(comm_mod_labels)):
    brain_map_key = 'pca_standardized_node_mean_' + comm_mod_labels[comm_mod]
    dist_dict = list(z_node_mean_norm_dist_dicts.values())[comm_mod]
    brain_maps[brain_map_key] = dist_dict['500_discov_wei']

for filename, brain_map in brain_maps.items():
    maps_path = pca_maps_path if 'pc' in filename else node_mean_maps_path
    brain_maps_split_path = os.path.join(maps_path, filename + '.svg')

    n = 108 if '125' in filename else 501
    #Switching the order of hemispheres for plotting
    switched_brain_map = brain_map.copy()
    switched_brain_map[:-n], switched_brain_map[-n:] = (brain_map[n:].copy(),
                                                        brain_map[:n].copy())

    cmap = 'RdBu_r' if 'pc' in filename else 'OrRd_r'
    if 'pc' in filename:
        vmin = -3
        vmax = 3
    elif '500' in filename and 'wei' in filename and 'geo' not in filename:
        vmin = 1.5
        vmax = 4.5
    elif '125_discov_wei' in filename:
        vmin = 1
        vmax = 2
    else:
        vmin = 1
        vmax = 3

    annot = cammoun125 if '125' in filename else cammoun500
    brain = plotting.plot_fsaverage(switched_brain_map, vmin = vmin,vmax = vmax,
                                    lhannot = annot.lh, rhannot = annot.rh,
                                    colormap = cmap, views = ['lat', 'med'],
                                    data_kws = {'representation': "wireframe"})
    brain.save_image(brain_maps_split_path)
    brain.close()

#Closeness centrality
closeness_maps_path = os.path.join(brain_maps_path, 'closeness')
make_dir(closeness_maps_path)

data_dicts = {'rand': node_mean_norm_dist_dicts,
              'geo_rand': node_mean_geo_norm_dist_dicts}
for null_mod, data_dict in data_dicts.items():
    comm_mod = 'shortest paths'
    key = '500_discov_wei'
    value = data_dict[comm_mod][key].copy()

    filename = null_mod + '_rank_difference.svg'
    closeness_maps_null_mod_path = os.path.join(closeness_maps_path, filename)

    empirical_closeness = np.reciprocal(np.nanmean(dist_dicts[comm_mod][key],
                                                   axis = 1))
    empirical_closeness_rank = rankdata(empirical_closeness, method = 'min')
    norm_closeness = np.reciprocal(value) #standardized closeness
    norm_closeness_rank = rankdata(norm_closeness, method = 'min')

    if null_mod == 'rand':
        closeness_rank_maps = {'empirical_closeness_rank':
                               empirical_closeness_rank,
                               'standardized_closeness_rank':
                               norm_closeness_rank}
        for closeness_key, brain_map in closeness_rank_maps.items():

            closeness_maps_split_path = os.path.join(closeness_maps_path,
                                                     closeness_key + '.svg')

            #Switching the order of hemispheres for plotting
            switched_brain_map = brain_map.copy()
            switched_brain_map[:-501], switched_brain_map[-501:] = (brain_map[501:].copy(),
                                                                    brain_map[:501].copy())

            data_kws = {'representation': "wireframe"}
            brain = plotting.plot_fsaverage(brain_map, vmin = 0, vmax = 1000,
                                            lhannot = cammoun500.lh,
                                            rhannot = cammoun500.rh,
                                            colormap = 'OrRd_r',
                                            views = ['lateral'],
                                            data_kws = data_kws)

            brain.save_image(closeness_maps_split_path)
            brain.close()

    closeness_rank_diff = empirical_closeness_rank - norm_closeness_rank

    closeness_rank_diff[:-501], closeness_rank_diff[-501:] = closeness_rank_diff[501:].copy(), closeness_rank_diff[:501].copy()

    brain = plotting.plot_fsaverage(closeness_rank_diff,
                                    lhannot = cammoun500.lh,
                                    rhannot = cammoun500.rh,
                                    colormap = 'RdBu_r',vmin = -300, vmax = 300,
                                    data_kws = data_kws, views = ['lat', 'med'])

    brain.save_image(closeness_maps_null_mod_path)
    brain.close()

#Edge-level plots
#-------------------------------------------------------------------------------
edge_plots_path = os.path.join(dir, 'edge-level plots')
make_dir(edge_plots_path)
covar_dicts = {'functional_connectivity': func_dict,
               'euclidean_distance': {'125': euc_dist125,
                                      '500': euc_dist500}}
data_dicts = {'rand': norm_dist_dicts, 'geo_rand': geo_norm_dist_dicts}
for comm_mod in comm_mods:
    edge_plots_comm_mod_path = os.path.join(edge_plots_path, comm_mod)
    make_dir(edge_plots_comm_mod_path)
    comm_mod_label = dyad_label if comm_mod == 'shortest paths' else 'PC1 score'
    for null_mod, data_dict in data_dicts.items():
        if comm_mod == 'pc1' and null_mod == 'geo_rand':
            continue
        edge_plots_null_mod_path = os.path.join(edge_plots_comm_mod_path,
                                                null_mod)
        make_dir(edge_plots_null_mod_path)
        edge_plots_txt_path = os.path.join(edge_plots_null_mod_path,
                                           'correlations.txt')
        for covar in covar_dicts.keys():

            mask = mask500

            key = '500_discov_wei'
            value = data_dict[comm_mod][key].copy()

            if covar == 'functional_connectivity':
                x, y = masking(value, comm_mod, mask = mask,
                               y = covar_dicts[covar][key[:-4]])
                xlabel, ylabel = comm_mod_label, 'functional connectivity'
            else:
                x, y = masking(covar_dicts[covar][key[:3]], comm_mod,
                               mask = mask, y = value)
                xlabel, ylabel = 'Euclidean distance (mm)', comm_mod_label

            _ = corr(x, y, edge_plots_txt_path, covar + '_' + key)

            edge_plots_split_path = os.path.join(edge_plots_null_mod_path,
                                                 covar + '_' + key + '.png')
            if comm_mod == 'pc1':
                ax = sns.histplot(x = x, y = y, bins = 100, cbar = True,
                                  cbar_kws = {'label': 'count',
                                              'orientation': 'vertical'})
                ax.set(xlabel = xlabel, ylabel = ylabel)
                ax.set_box_aspect(1)
            else: ax = histplot(x, y, xlabel, ylabel)
            sns.despine(ax = ax)
            save_plot(ax, edge_plots_split_path)

            if comm_mod == 'shortest paths' and null_mod == 'rand':
                #Coloring based on the partition between
                #monosynaptic and polysynaptic dyads
                filename = '{}_{}_partition_conn.png'.format(covar, key)
                partition_conn_split_path=os.path.join(edge_plots_null_mod_path,
                                                       filename)
                hue = struct_den_dict[key].copy()
                #Mapping monosynaptic dyads to 1 and polysynaptic dyads to 0
                hue[hue > 0] = 1
                hue, _ = masking(hue, comm_mod, mask = mask)
                ax = histplot(x, y, xlabel, ylabel,
                              hue = hue, hue_order = [1, 0],
                              palette = ['orange', 'royalblue'])
                legend = ax.get_legend()
                handles = legend.legendHandles
                legend.remove()
                ax.legend(handles, ['monosynaptic', 'polysynaptic'],
                          bbox_to_anchor = (0, 1.01, 1, 0.1), loc ='lower left')
                sns.despine(ax = ax)
                save_plot(ax, partition_conn_split_path)

                #Exponential curve fitting and plotting
                exp_split_path = os.path.join(edge_plots_null_mod_path,
                                              covar + '_' + key + '_exp.png')
                ax = histplot(x, y, xlabel, ylabel)
                sns.despine(ax = ax)
                if covar == 'euclidean_distance':
                    p0 = (-1, -1, ax.get_ylim()[1])
                    exp_curve_fit(x, y, p0, ax, exp_split_path)
                elif covar == 'functional_connectivity':
                    p0 = (ax.get_ylim()[1], 1)
                    exp_curve_fit(x, y, p0, ax, exp_split_path, decay = True)

            #Plotting PC1 scores of strictly polysynaptic dyads
            if comm_mod == 'pc1':
                poly_split_path = os.path.join(edge_plots_null_mod_path,
                                               covar + '_' + key + '_poly.png')

                mono_idx = struct_den_dict[key].copy()
                #Mapping monosynaptic dyads to 1 and polysynaptic dyads to 0
                mono_idx[mono_idx > 0] = 1
                #Creating mask
                mono_idx = np.nonzero(mono_idx)
                #Taking out monosynaptic dyads
                value[mono_idx] = np.nan

                if covar == 'functional_connectivity':
                    x, y = masking(value, comm_mod, mask = mask,
                                   y = covar_dicts[covar][key[:-4]])
                else:
                    x, y = masking(covar_dicts[covar][key[:3]], comm_mod,
                                   mask = mask, y = value)

                _ = corr(x, y, edge_plots_txt_path, covar + '_' + key + '_poly')

                ax = sns.histplot(x = x, y = y, bins = 100, cbar = True,
                                  cbar_kws = {'label': 'count',
                                              'orientation': 'vertical'})
                ax.set(xlabel = xlabel, ylabel = ylabel)
                ax.set_box_aspect(1)
                sns.despine(ax = ax)
                save_plot(ax, poly_split_path)

                if covar == 'euclidean_distance':
                    #Exponential curve fitting and plotting
                    filename = covar + '_' + key + '_poly_exp.png'
                    exp_split_path = os.path.join(edge_plots_null_mod_path,
                                                  filename)
                    ax = sns.histplot(x = x, y = y, bins = 100, cbar = True,
                                      cbar_kws = {'label': 'count',
                                                  'orientation': 'vertical'})
                    ax.set(xlabel = xlabel, ylabel = ylabel)
                    ax.set_box_aspect(1)
                    sns.despine(ax = ax)
                    p0 = (-1, -1, ax.get_ylim()[1])
                    exp_curve_fit(x, y, p0, ax, exp_split_path)

#Generating Hungarian spin nulls
seed = 0
hungarian_nulls = {}

for scale in '125', '500':

    annot = cammoun125 if scale == '125' else cammoun500
    coords, hemi = freesurfer.find_parcel_centroids(lhannot = annot.lh,
                                                    rhannot = annot.rh)
    spins = stats.gen_spinsamples(coords, hemi, method = 'hungarian',
                                  seed = seed, verbose = True,
                                  return_cost = False)

    n = 108 if scale == '125' else 501
    #Switching the order of hemispheres to fit the data
    spins[:n], spins[n:]= spins[-n:].copy() -len(spins) +n, spins[:-n].copy() +n

    hungarian_nulls[scale] = spins

#Edge-level partition specificity
#-------------------------------------------------------------------------------
partition_path = os.path.join(dir, 'edge-level partition specificity')
make_dir(partition_path)

#7 Yeo networks
partition_yeo_7_path = os.path.join(partition_path, 'Yeo 7')
make_dir(partition_yeo_7_path)
data_dicts = {'rand': norm_dist_dicts, 'geo_rand': geo_norm_dist_dicts}
for comm_mod in comm_mods:
    comm_mod_path = os.path.join(partition_yeo_7_path, comm_mod)
    make_dir(comm_mod_path)
    for null_mod, data_dict in data_dicts.items():
        if comm_mod == 'pc1' and null_mod == 'geo_rand':
            continue
        null_mod_path = os.path.join(comm_mod_path, null_mod)
        make_dir(null_mod_path)
        for key, value in data_dict[comm_mod].items():
            for dyads in ['all', 'poly']:
                if dyads == 'poly' and key != '500_discov_wei':
                    continue

                #symmetrise PC1 matrix
                if comm_mod == 'pc1':
                    value = (value + value.T)/2

                filename = 'cross_networks_heatmap_' +key+ '_' +dyads+ '.svg'
                heatmap_split_path = os.path.join(null_mod_path, filename)
                filename = 'signif_diff_heatmap_' + key + '_' + dyads + '.svg'
                signif_diff_heatmap_split_path = os.path.join(null_mod_path,
                                                              filename)

                res = 125 if '125' in key else 500
                null_scale = '125' if '125' in key else '500'

                nodes = 219 if res == 125 else 1000
                idx = partition_yeo_7_dict['idx'][res]
                rsn_idx = partition_yeo_7_dict['rsn_idx'][res]
                yeo_7 = partition_yeo_7_dict['yeo_7'][res]
                yeo_7_labels = np.unique(yeo_7)

                if dyads == 'poly':
                    value = value.copy()
                    mono_idx = struct_den_dict[key].copy()
                    #Mapping monosynaptic dyads to 1 and polysynaptic dyads to 0
                    mono_idx[mono_idx > 0] = 1
                    #Creating mask
                    mono_idx = np.nonzero(mono_idx)
                    #Taking out monosynaptic dyads
                    value[mono_idx] = np.nan
                #Reordering by Yeo network affiliation
                ordered_val = value[idx][:, idx]

                #dictionary columns/Yeo network labels
                columns = np.array(['dm', 'da', 'fp', 'lim', 'sm', 'va', 'vis'])
                #dictionary of intra-network standardized shortest path lengths
                empirical_dict = {}
                #7x7 matrix for partitioning intra and inter-Yeo networks dyads
                matrix = np.zeros((7, 7))
                #Populating the matrix
                for i in range(7):
                    within_slice = slice(rsn_idx[i], rsn_idx[i + 1])
                    #intra-network standardized shortest path lengths
                    rsn_val = ordered_val[within_slice, within_slice]
                    empirical_dict[columns[i]] = rsn_val.flatten()
                    for j in range(7):
                        #ith network indices
                        i_slice = slice(rsn_idx[i], rsn_idx[i + 1])
                        #jth network indices
                        j_slice = slice(rsn_idx[j], rsn_idx[j + 1])
                        #standardized shortest path lengths
                        #between networks i and j
                        slice_val = ordered_val[i_slice, j_slice]
                        #mean standardized path length
                        matrix[i, j] = np.nanmean(slice_val)
                #Reordering the matrix based on network affiliation
                matrix = matrix[nets_reorder_idx]
                matrix = matrix[:, nets_reorder_idx]

                #Plotting the matrix
                if comm_mod == 'pc1':
                    label = 'PC1 score'
                else: label = 'mean standardized shortest path length'
                if (null_mod == 'geo_rand' or
                    comm_mod == 'shortest paths' and dyads == 'poly'):
                    cmap = 'magma'
                    center = None
                else:
                    cmap = 'RdBu_r'
                    center = 0
                fig, ax = plt.subplots(1, figsize = (10, 10))
                sns.heatmap(matrix, ax = ax, cbar_kws = {'label': label},
                            cmap = cmap, center = center, square = True,
                            xticklabels = columns[nets_reorder_idx],
                            yticklabels = columns[nets_reorder_idx],
                            mask = np.triu(np.ones((7, 7)), k = 1))

                fig.tight_layout()
                fig.savefig(heatmap_split_path, dpi = 300)
                plt.close(fig)

                if comm_mod == 'shortest paths' and key == '500_discov_wei':
                    #matrix of differences of
                    #intra-network mean standardized shortest path lengths
                    diff_matrix = np.zeros((7, 7))
                    #matrix indicating significant differences
                    signif_matrix = np.zeros((7, 7))
                    #pairs of indices of upper triangle
                    triu_idx = list(zip(*np.triu_indices(7, k = 1)))
                    #Populating differences matrix for each pair of networks
                    for x_idx, y_idx in triu_idx:

                        #intra-network mean standardized shortest path lengths
                        #of networks x and y
                        x, y = empirical_dict[columns[nets_reorder_idx][x_idx]], empirical_dict[columns[nets_reorder_idx][y_idx]]

                        #difference of the means
                        diff = np.nanmean(y) - np.nanmean(x)
                        diff_matrix[y_idx, x_idx] = diff

                        #spin null means' differences
                        x_label = yeo_7_labels[nets_reorder_idx][x_idx]
                        y_label = yeo_7_labels[nets_reorder_idx][y_idx]
                        null_means_diffs = Parallel(n_jobs=4)(delayed(nets_null_means_diff)(yeo_7, null_idx, nodes, x_label, y_label, ordered_val) for null_idx in hungarian_nulls[null_scale].T)
                        null_means_diffs = np.array(null_means_diffs)

                        suffix = "_{}_{}_{}".format(x_label, y_label, dyads)
                        p_spin = spin_tests(null_mod_path, key + suffix,
                                            diff, corr = False,
                                            null_distrib = null_means_diffs)

                        #Bonferroni-corrected significance threshold
                        bonferroni_p = 0.05/((7**2 - 7)/2)
                        #Populating the significant differences matrix
                        if p_spin < bonferroni_p:
                            signif_matrix[y_idx, x_idx] = 1

                    #Plotting the differences matrix
                    fig, ax = plt.subplots(1, figsize = (10, 10))
                    cbar_kws={'label': 'difference of the means (row - column)'}
                    sns.heatmap(diff_matrix, ax = ax, cbar_kws = cbar_kws,
                                cmap = 'RdBu_r', center = 0, square = True,
                                xticklabels = columns[nets_reorder_idx],
                                yticklabels = columns[nets_reorder_idx],
                                mask = np.triu(np.ones((7, 7))))

                    #Outlining the significant cells in purple
                    for x_idx, y_idx in triu_idx:
                        if signif_matrix[y_idx, x_idx] == 1:
                            ax.add_patch(Rectangle((x_idx, y_idx), 1, 1,
                                                   fill = False, lw = 3,
                                                   edgecolor = 'mediumpurple'))

                    fig.tight_layout()
                    fig.savefig(signif_diff_heatmap_split_path, dpi = 300)
                    plt.close(fig)

#within-between 7 Yeo networks
partition_w_b_path = os.path.join(partition_path, 'within-between Yeo 7')
make_dir(partition_w_b_path)
data_dicts = {'rand': norm_dist_dicts, 'geo_rand': geo_norm_dist_dicts}
for comm_mod in comm_mods:
    comm_mod_path = os.path.join(partition_w_b_path, comm_mod)
    make_dir(comm_mod_path)
    for null_mod, data_dict in data_dicts.items():
        if comm_mod == 'pc1' and null_mod == 'geo_rand':
            continue
        null_mod_path = os.path.join(comm_mod_path, null_mod)
        make_dir(null_mod_path)
        pval_txt_path = os.path.join(null_mod_path, 'p_spins.txt')
        diff_txt_path = os.path.join(null_mod_path, 'difference.txt')
        for key, value in data_dict[comm_mod].items():
            for dyads in ['all', 'poly']:
                if dyads == 'poly' and key != '500_discov_wei':
                    continue

                filename = key + '_' + dyads + '.svg'
                split_path = os.path.join(null_mod_path, filename)

                res = 125 if '125' in key else 500
                null_scale = '125' if '125' in key else '500'

                nodes = 219 if res == 125 else 1000
                idx = partition_yeo_7_dict['idx'][res]
                rsn_idx = partition_yeo_7_dict['rsn_idx'][res]
                yeo_7 = partition_yeo_7_dict['yeo_7'][res]

                if dyads == 'poly':
                    value = value.copy()
                    mono_idx = struct_den_dict[key].copy()
                    #Mapping monosynaptic dyads to 1 and polysynaptic dyads to 0
                    mono_idx[mono_idx > 0] = 1
                    #Creating mask
                    mono_idx = np.nonzero(mono_idx)
                    #Taking out monosynaptic dyads
                    value[mono_idx] = np.nan
                #Reordering by Yeo network affiliation
                ordered_val = value[idx][:, idx]
                #Extracting within/between-networks distributions and
                #empirical difference of the means
                within_distrib, between_distrib, empirical = partition_w_b(rsn_idx, ordered_val)

                #spin null means' differences
                null_means_diffs = []
                #rotated indices
                for null_idx in hungarian_nulls[null_scale].T:

                    permuted_labels = yeo_7[null_idx]
                    #Mask of permuted within vs between-networks dyads
                    permuted_partition = np.zeros((nodes, nodes))
                    for i in range(nodes):
                        for j in range(nodes):
                            #within-networks
                            if permuted_labels[i] == permuted_labels[j]:
                                permuted_partition[i, j] = 1
                            #between-networks
                            else:
                                permuted_partition[i, j] = 0

                    #indices of null within-networks dyads
                    null_within_idx = np.where(permuted_partition == 1)
                    #mean of null within-networks dyads
                    null_within_mean = np.nanmean(ordered_val[null_within_idx])
                    #indices of null between-networks dyads
                    null_between_idx = np.where(permuted_partition == 0)
                    #mean of null between-networks dyads
                    null_between_mean= np.nanmean(ordered_val[null_between_idx])

                    #null difference of the means
                    null_means_diffs.append(null_between_mean -null_within_mean)

                null_means_diffs = np.array(null_means_diffs)


                p_spin = spin_tests(null_mod_path, key + '_' + dyads,
                                    empirical, corr = False,
                                    null_distrib = null_means_diffs)

                #Writing differences of the means and p-values
                means_diff_txt='difference = {}\n'.format(np.around(empirical,
                                                                    decimals=2))
                with open(diff_txt_path, 'a') as means_diff_file:
                    content = key + '_' + dyads + ': ' + means_diff_txt
                    means_diff_file.write(content)

                pval_txt = 'p-value = {}\n'.format(np.around(p_spin,decimals=3))
                with open(pval_txt_path, 'a') as pvals_file:
                    content = key + '_' + dyads + ': ' + pval_txt
                    pvals_file.write(content)

                #Plotting the empirical within and between-networks distribution
                ax = sns.violinplot(data = [within_distrib, between_distrib],
                                    cut = 0, inner = 'quartile', orient = 'v',
                                    palette = ['royalblue', 'firebrick'])

                #Plotting asterisk if significant
                max_val = max(np.nanmax(within_distrib),
                              np.nanmax(between_distrib))
                ax.text(0.5, max_val, '*' if p_spin < 0.05 else 'ns',
                        ha = 'center', va = 'bottom')

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['within network', 'between network'])
                ax.set_ylabel(dyad_label)
                sns.despine(ax = ax, trim = True)
                save_plot(ax, split_path)

#Node-level plots
#-------------------------------------------------------------------------------
node_plots_path = os.path.join(dir, 'node-level plots')
make_dir(node_plots_path)

node_label = 'node-wise mean standardized shortest path length'
x_dicts = {'weighted degree': strengths_dict,
           'log betweenness centrality': betweenness_dict,
           'participation coefficient': participation_dict}
for comm_mod in comm_mods:
    node_plots_comm_mod_path = os.path.join(node_plots_path, comm_mod)
    make_dir(node_plots_comm_mod_path)
    comm_mod_label = node_label if comm_mod == 'shortest paths' else 'PC1 score'
    node_plots_txt_path = os.path.join(node_plots_comm_mod_path,
                                       'correlations.txt')
    for centrality in x_dicts.keys():

        key = '500_discov_wei'
        value = node_mean_norm_dist_dicts[comm_mod][key].copy()

        x, y = x_dicts[centrality][key].copy(), value
        inf = np.isfinite(x)
        if centrality == 'log betweenness centrality': x[~inf] = np.nan

        spearman_r = corr(x, y, node_plots_txt_path, centrality + '_' + key)
        #Plotting nodal bivariate histogram
        node_plots_split_path = os.path.join(node_plots_comm_mod_path,
                                             centrality + '_' + key + '.svg')
        ax = histplot(x, y, centrality, comm_mod_label)
        sns.despine(ax = ax)
        save_plot(ax, node_plots_split_path)

        _ = spin_tests(node_plots_comm_mod_path, centrality + '_' + key,
                       spearman_r[0], x = x, y = y, nulls_idx = hungarian_nulls)

#Neurosynth analysis
#-------------------------------------------------------------------------------
cammoun_ns_dir = '../../data/original_data/cammoun_ns/'
#Dataframes of Cammoun parcel-wise probabilistic measures
#for 123 Cognitive atlas terms
cammoun_ns_125_df = pd.read_csv(cammoun_ns_dir + 'scale125.csv', index_col = 0)
cammoun_ns_500_df = pd.read_csv(cammoun_ns_dir + 'scale500.csv', index_col = 0)

cammoun_ns_df_idx = pickle_load('cammoun_ns_df_idx')

ns_path = os.path.join(dir, 'neurosynth')
make_dir(ns_path)
for comm_mod in comm_mods:
    ns_comm_mod_path = os.path.join(ns_path, comm_mod)
    make_dir(ns_comm_mod_path)
    data_dicts = {'rand': node_mean_norm_dist_dicts,
                  'geo_rand': node_mean_geo_norm_dist_dicts,
                  'poly_rand': poly_node_mean_dicts['rand'],
                  'poly_geo_rand': poly_node_mean_dicts['geo_rand']}
    for null_mod, data_dict in data_dicts.items():
        if comm_mod == 'pc1' and null_mod != 'rand':
            continue
        ns_null_mod_path = os.path.join(ns_comm_mod_path, null_mod)
        make_dir(ns_null_mod_path)
        for key, value in data_dict[comm_mod].items():
            split_path = os.path.join(ns_null_mod_path, key + '.svg')

            if '125' in key:
                df = cammoun_ns_125_df
                df_idx = cammoun_ns_df_idx['scale125']
                null_scale = '125'
            else:
                df = cammoun_ns_500_df
                df_idx = cammoun_ns_df_idx['scale500']
                null_scale = '500'

            #correlations/p-values between nodal standardized path lengths maps
            #and Neurosynth maps
            corrs = []
            p_spins = []
            #null nodal standardized path lengths maps
            nulls = value[hungarian_nulls[null_scale]].T
            nulls = nulls.copy()
            #spin null distributions of Spearman correlations across
            #null nodal standardized path lengths maps and
            #all 123 functional activation maps
            null_distributions = np.zeros((len(nulls), len(df.columns)))
            #Neurosynth map index
            for column in range(len(df.columns)):
                #nodal standardized path lengths map and Neurosynth map
                x, y = value.copy(), df.loc[df_idx][df.columns[column]].values
                corrs.append(spearmanr(x, y, nan_policy = 'omit')[0])
                for null in range(len(nulls)):
                    x = nulls[null]
                    corr = spearmanr(x, y, nan_policy = 'omit')[0]
                    null_distributions[null, column] = corr
                corr = corrs[column]
                null_distribution = null_distributions[:, column]
                null_distribution_mean = np.mean(null_distribution)
                demeaned_null_distribution = (null_distribution -
                                              null_distribution_mean)
                demeaned_corr = corr - null_distribution_mean
                #number of more extreme null means
                p_sum = (np.abs(demeaned_null_distribution) >=
                         np.abs(demeaned_corr)).sum()
                #proportion of more extreme null means
                p_spin = p_sum/len(demeaned_null_distribution)
                p_spins.append(p_spin)
            corrs = np.array(corrs)
            p_spins = np.array(p_spins)

            #p-values passing the significance threshold
            signif_cond = p_spins < 0.05
            signif_corrs = corrs[signif_cond]
            if np.any(signif_corrs):

                #reordering indices for sorting significant correlations/terms
                #by correlation magnitude
                reorder_idx = np.argsort(signif_corrs)
                ordered_signif_corrs = signif_corrs[reorder_idx]
                ordered_signif_p_spins = p_spins[signif_cond][reorder_idx]
                ordered_signif_terms = df.columns[signif_cond][reorder_idx]

                #Populating the palette according to
                #alpha = 0.05 Bonferroni-corrected threshold
                palette = []
                for term in range(len(ordered_signif_terms)):
                    #significant
                    if ordered_signif_p_spins[term] < 0.05/len(df.columns):
                        palette.append('teal')
                    else: palette.append('grey')

                #Plotting the barplot of significant correlations
                ax = sns.barplot(x = ordered_signif_terms,
                                 y = ordered_signif_corrs,
                                 palette = palette)
                plt.xticks(rotation = 45, horizontalalignment = 'right')
                ax.set_xlabel('Neurosynth terms')
                ax.set_ylabel("Spearman's r")
                sns.despine(ax = ax, bottom = True)
                save_plot(ax, split_path)
