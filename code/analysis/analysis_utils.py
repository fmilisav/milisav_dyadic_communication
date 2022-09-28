import numpy as np
from scipy.stats import zscore, spearmanr
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import seaborn as sns
sns.set_style("ticks")
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import plotly.graph_objects as go

from netneurotools import datasets

import pandas as pd

import os
import sys
sys.path.append('..')
from utils import make_dir


def masking(x, comm_mod, mask = None, y = None):

    #Extracting full asymmetric communication matrices
    if (comm_mod == 'mean first-passage time' or
        comm_mod == 'search information'):
        di = np.diag_indices(len(x))
        x = x.astype(float)
        x[di] = np.nan
        x = x.flatten()
        if y is not None:
            y = y.astype(float)
            y[di] = np.nan
            y = y.flatten()
    #Extracting the upper triangle for symmetric communication matrices
    else:
        x = x[mask]
        if y is not None: y = y[mask]

    return x, y

#function for creating a bivariate histogram (empirical vs rewired)
def bivar_hists(x, y, **kwargs):

    marginal_kws = {'element': 'step', 'alpha': 0.7}
    cbar_kws_dict = {'label': 'count', 'orientation': 'horizontal', 'pad': 0.1}

    g = sns.jointplot(x = x, y = y, kind = 'hist',
                      height = 10, ratio = 7,
                      marginal_kws = marginal_kws,
                      cbar_kws = cbar_kws_dict,
                      cbar = True, bins = 'sqrt',
                      **kwargs)
    g.ax_joint.set(xlabel = 'empirical', ylabel = 'rewired')

    return g

#after plotting identity line
#function for setting bivariate histogram's limits
def set_joint_plot_lims(g, lims):

    g.ax_joint.set_xlim(lims)
    g.ax_joint.set_ylim(lims)
    g.ax_joint.set_aspect('equal')

#function for setting bivariate histogram's subplots' positions
def set_joint_plot_pos(g):

    joint_pos = g.ax_joint.get_position()
    marg_x_pos = g.ax_marg_x.get_position()
    marg_y_pos = g.ax_marg_y.get_position()
    g.ax_marg_x.set_position([joint_pos.x0, marg_x_pos.y0,
                              joint_pos.width, marg_x_pos.height])
    marg_y_x0 = joint_pos.x1 + marg_x_pos.y0 - joint_pos.y1
    g.ax_marg_y.set_position([marg_y_x0, joint_pos.y0,
                              marg_y_pos.width, joint_pos.height])
    cbar_pos = g.fig.axes[-1].get_position()
    g.fig.axes[-1].set_position([joint_pos.x0, cbar_pos.y0,
                                 joint_pos.width, cbar_pos.height])

def save_plot(ax, path):

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi = 300)
    plt.close(fig)

#function for calculating and writing Spearman correlation and p-value
def corr(x, y, txt_path, key):

    spearman_r = list(spearmanr(x, y, nan_policy = 'omit'))
    spearman_r[0] = np.around(spearman_r[0], decimals = 2)
    with open(txt_path, 'a') as corr_file:
        results = ("Spearman's r = {}, p-value = {}\n").format(spearman_r[0],
                                                               spearman_r[1])
        corr_file.write(key + ': ' + results)

    return spearman_r

#function for generating a null distribution of Spearman correlations
def corr_nulls(x, y, key, nulls_idx):

    scale = '125' if '125' in key else '500'
    null_distribution = []
    nulls = y[nulls_idx[scale]].T
    for null in nulls:
        corr = spearmanr(x, null, nan_policy = 'omit')[0]
        null_distribution.append(corr)
    null_distribution = np.array(null_distribution)

    return null_distribution

#function for calculating p-spin and saving a spin test plot
def spin_tests(parent_path, key, empirical, corr = True,
               x = None, y = None, nulls_idx = None, null_distrib = None):

    #spin test plot path
    path = '_Spearman_r_spin_tests.svg' if corr else '_spin_tests.svg'
    spin_tests_split_path = os.path.join(parent_path, key + path)

    #generating a null distribution of Spearman correlations
    if corr:
        null_distrib = corr_nulls(x, y, key, nulls_idx)

    #calculating p-spin
    null_distrib_mean = np.mean(null_distrib)
    demeaned_null_distrib = null_distrib - null_distrib_mean
    demeaned_empirical = empirical - null_distrib_mean
    p_sum=(np.abs(demeaned_null_distrib)>=np.abs(demeaned_empirical)).sum()
    p_spin = p_sum/len(demeaned_null_distrib)

    #plotting spin test plot
    ax = sns.kdeplot(x = null_distrib, cut = 0)
    ax.axvline(x = empirical, c = 'orange')
    ax.legend(['spin null distribution',
               'empirical, p = {}'.format(np.around(p_spin, decimals = 3))],
               bbox_to_anchor = (0, 1.01, 1, 0.1), loc = 'lower left')
    if corr: ax.set(xlabel = "Spearman's r")
    else: ax.set(xlabel = 'difference of the means')
    ax.set_box_aspect(1)
    sns.despine(ax = ax)
    save_plot(ax, spin_tests_split_path)

    return p_spin

#function for plotting custom histograms
def histplot(x, y, xlabel, ylabel, **kwargs):

    ax = sns.histplot(x = x, y = y, bins = 'sqrt', cbar = True,
                      cbar_kws = {'label': 'count', 'orientation': 'vertical'},
                      **kwargs)
    ax.set(xlabel = xlabel, ylabel = ylabel)
    ax.set_box_aspect(1)

    return ax

#exponential function
def exp(x, n, l, c):
    return n * np.exp(l * x) + c

#exponential decay function
def expDecay(x, n, l):
    return n * np.exp(-l * x)

#function for fitting exponential curves
def exp_curve_fit(x, y, p0, ax, path, decay = False):

    #exponential function
    exp_func = exp if decay == False else expDecay

    #indices of nan values in either arrays
    nan = np.isnan(x) | np.isnan(y)
    x, y = x[~nan], y[~nan]

    #least-squares fitting yields
    #optimal parameter values and covariance estimates
    popt, pcov = curve_fit(exp_func, x, y, p0, maxfev = 10000)

    sortedX = np.sort(x)
    #evenly spaced evaluation points over the interval of x for plotting
    x_linspace = np.linspace(min(x), max(x))
    if decay == True:
        #exponential decay function fitted parameters
        n, l = popt
        #predicted y
        y_pred = exp_func(x, n, l)
        #predicted y over evaluation points
        linspace_y_pred = exp_func(x_linspace, n, l)
    else:
        #exponential function fitted parameters
        n, l, c = popt
        #predicted y
        y_pred = exp_func(x, n, l, c)
        #predicted y over evaluation points
        linspace_y_pred = exp_func(x_linspace, n, l, c)

    #sum of squares residuals
    ssRes = np.square(y - y_pred)
    #total sum of squares
    ssTot = np.square(y - np.mean(y))
    r2 = 1 - np.sum(ssRes)/np.sum(ssTot)

    #Plotting the fitted curve and its function
    if decay == True:
        label = ('Y = {:.2f}e^(-{:.2f}x)\n'
                 'R2 = {:.2f}').format(n, l, r2)
    else:
        label = ('Y = {:.2f}e^({:.2f}x) + {:.2f}\n'
                 'R2 = {:.2f}').format(n, l, c, r2)
    ax.plot(x_linspace, linspace_y_pred,
            'k-', label = label)
    ax.legend(bbox_to_anchor = (0, 1.01, 1, 0.1),
              loc = 'lower left')
    save_plot(ax, path)

nets_reorder_idx = [6, 4, 1, 5, 3, 2, 0] #reordering index for Yeo networks
cammoun_info = pd.read_csv(datasets.fetch_cammoun2012()['info'])
#function returning useful data for partition specificty analyses among
#Yeo networks
def partition_yeo_7(res, df = cammoun_info,nets_reorder_idx = nets_reorder_idx):

    scale = 'scale125' if res == 125 else 'scale500'

    nodes = 219 if res == 125 else 1000

    df_rows_bool_arr=(df['structure'] == 'cortex') & (df['scale'] == scale)
    df.loc[df_rows_bool_arr, 'idx'] = range(nodes)

    #indices for reordering regions by Yeo network affiliation and
    #original order within Yeo blocks
    idx = df.loc[df_rows_bool_arr].sort_values(['yeo_7',
                                                'idx']).idx.values.astype(int)
    #corresponding Yeo labels across regions
    yeo_7 = df.loc[df_rows_bool_arr].sort_values('yeo_7')['yeo_7'].values

    #indices delimiting each block
    rsn_idx = [(yeo_7 == rsn).nonzero()[0][0] for rsn in np.unique(yeo_7)]
    rsn_idx.append(nodes)

    #indices for reordering regions by ordered Yeo network affiliation
    nets = np.unique(yeo_7)
    reorder_idx = []
    for net in nets[nets_reorder_idx]:
        reorder_idx.extend(np.where(yeo_7 == net)[0])

    return idx, yeo_7, rsn_idx, reorder_idx

#function for computing null difference of the means
def nets_null_means_diff(labels, null_idx, nodes, x_label, y_label,
                         ordered_val):

    permuted_labels = labels[null_idx]
    permuted_partition = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in range(nodes):
            #node pair belonging to network x
            if (permuted_labels[i] == x_label and
                permuted_labels[j] == x_label):
                permuted_partition[i, j] = 1
            #node pair belonging to network x
            elif (permuted_labels[i] == y_label and
                  permuted_labels[j] == y_label):
                permuted_partition[i, j] = 2
            #other node pairs
            else:
                permuted_partition[i, j] = 0

    #indices of node pairs belonging to network x
    null_x_idx = np.where(permuted_partition == 1)
    #null mean standardized communication distance
    #across node pairs belonging to network x
    null_x_mean = np.nanmean(ordered_val[null_x_idx])
    #indices of node pairs belonging to network y
    null_y_idx = np.where(permuted_partition == 2)
    #null mean standardized communication distance
    #across node pairs belonging to network y
    null_y_mean = np.nanmean(ordered_val[null_y_idx])

    #null difference of the means
    null_means_diff = null_x_mean - null_y_mean
    return null_means_diff

#function for partitioning dyads within and between Yeo networks and
#computing the difference of the means
def partition_w_b(rsn_idx, ordered_val):

    within_distrib = []
    between_distrib = []
    for i in range(7):
        #indices of within-network dyads
        within_slice = slice(rsn_idx[i], rsn_idx[i + 1])
        #within-network values
        rsn_val = ordered_val[within_slice, within_slice]
        #between-network values
        between_val_l = ordered_val[within_slice, :rsn_idx[i]]
        between_val_r = ordered_val[within_slice, rsn_idx[i + 1]:]
        within_distrib.extend(rsn_val.flatten())
        between_distrib.extend(between_val_l.flatten())
        between_distrib.extend(between_val_r.flatten())

    #means' difference
    means_diff = np.nanmean(between_distrib) - np.nanmean(within_distrib)

    return within_distrib, between_distrib, means_diff

#function for standardizing the columns of the mean first-passage time matrix
#to remove nodal bias
def mfpt_zscore(dist_dicts):
    for key, value in dist_dicts['mean first-passage time'].items():
        dist_dicts['mean first-passage time'][key] = zscore(value, ddof = 1,
                                                            nan_policy = 'omit')

    return dist_dicts

#function for enforcing symmetry of communication matrices and
#recomputing regional communication distance
def symmetrise(node_mean_dist_dicts, dist_dicts):
    for comm_mod, node_mean_dist_dict in node_mean_dist_dicts.items():
        if (comm_mod == 'search information' or
            comm_mod == 'mean first-passage time'):
            for key, value in node_mean_dist_dict.items():
                dist = dist_dicts[comm_mod][key]
                symm_dist = (dist + dist.T)/2

                node_mean_dist_dicts[comm_mod][key] = np.nanmean(symm_dist,
                                                                 axis = 1)

#function for standardizing communication matrices or arrays
def zscore_dist_dicts(dist_dicts, level):
    for comm_mod, dist_dict in dist_dicts.items():
        for key, value in dist_dict.items():
            if level == 'edge':
                dist_dicts[comm_mod][key] = zscore(value, axis = None, ddof = 1,
                                                   nan_policy = 'omit')
            else:
                dist_dicts[comm_mod][key] = zscore(value, ddof = 1,
                                                   nan_policy = 'omit')
    return dist_dicts

#function for applying PCA to communication matrices

#considered communication models
pca_comm_mods = ['shortest paths', 'search information', 'path transitivity',
                 'communicability', 'mean first-passage time']
def pca_dist_dicts(dist_dicts, level, dir, pca_comm_mods = pca_comm_mods):

    #Building X
    X_dict = {}
    key = '500_discov_wei'
    X = []
    for dist_dict in dist_dicts.values():
        if level == 'edge':
            dist = dist_dict[key]
            di = np.diag_indices(len(dist))
            dist[di] = np.nan
            dist = dist.flatten()
            #Excluding undefined diagonal elements
            dist = dist[~np.isnan(dist)]
            X.append(dist)
        else:
            X.append(dist_dict[key])
    X_dict[key] = np.array(X).T

    pca_scores_dict = {'pc1': {}}
    for key, value in X_dict.items():

        n = 219 if '125' in key else 1000

        pca = PCA(svd_solver = 'full')

        if level == 'edge':
            #PC1 scores
            pc1 = -pca.fit_transform(value)[:, 0]
            #Reinserting undefined diagonal elements
            for i in range(n - 1):
                pc1 = np.insert(pc1, n*i + i, np.nan)
            pc1 = np.append(pc1, np.nan)
            #Reshaping the aggregate communication matrix
            pca_scores_dict['pc1'][key] = pc1.reshape((n, n))
        else:
            #PC1 scores
            pc1 = pca.fit_transform(value)[:, 0]
            pca_scores_dict['pc1'][key] = np.squeeze(pc1)

        #Plotting PC1 loadings and explained variance ratios across components
        if key == '500_discov_wei':

            pca_path = os.path.join(dir, 'PCA')
            make_dir(pca_path)

            components_file = '{}_level_pca_components.svg'.format(level)
            expl_vari_file = '{}_level_pca_explained_variance.svg'.format(level)
            pca_components_path = os.path.join(pca_path, components_file)
            pca_expl_vari_path = os.path.join(pca_path, expl_vari_file)

            fig = go.Figure()

            #origin
            fig.add_trace(go.Scatterpolar(
                r=[0]*360,
                theta=list(range(360)),
                line = dict(smoothing = 1,
                            color = "black",
                            shape = "spline"),
                mode = 'lines',
                showlegend = False
            ))

            if level == 'edge':
                #PC1 loadings
                pc1_sqrt_vari = np.sqrt(pca.explained_variance_[0])
                pc1_loadings = -pca.components_[0] * pc1_sqrt_vari
                #shortest paths loading
                sp_loading = -pca.components_[0][0] * pc1_sqrt_vari
            else:
                #PC1 loadings
                pc1_sqrt_vari = np.sqrt(pca.explained_variance_[0])
                pc1_loadings = pca.components_[0] * pc1_sqrt_vari
                #shortest paths loading
                sp_loading = pca.components_[0][0] * pc1_sqrt_vari
            r = np.append(pc1_loadings, sp_loading)
            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=[0, 72, 144, 216, 288, 360],
                fill='toself',
                marker = dict(color = '#636EFA'),
                line = dict(color = '#636EFA'),
                name='PC1'
            ))

            #labels
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True
                    ),
                    angularaxis = dict(visible = True,
                                       type = "category",
                                       ticktext = pca_comm_mods,
                                       tickvals = [0, 72, 144, 216, 288])
                    ),
                showlegend=True
            )

            fig.write_image(pca_components_path)

            #explained variance ratios
            components = np.array(range(len(pca_comm_mods))) + 1
            ax = sns.scatterplot(x = components,
                                 y = pca.explained_variance_ratio_,
                                 color = 'grey')
            ax.set_xticks(components)
            ax.set(xlabel = 'component',
                   ylabel = 'variance explained ratio')
            ax.set_box_aspect(1)
            fig = ax.get_figure()
            sns.despine(fig)
            fig.tight_layout()
            fig.savefig(pca_expl_vari_path, bbox_inches='tight',
                        dpi = 300)
            plt.close(fig)

    return pca_scores_dict
