#!/usr/bin/env python

'''
This file includes modules that help with visualization
===============================================
Author: Gholamali Rahnavard (gholamali.rahnavard@gmail.com)
'''
# from numpy import array, median
import argparse
import math
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
# activate latex text rendering
# rc('text', usetex=True)
# matplotlib.rcParams['text.usetex']=True
# matplotlib.rcParams['text.latex.unicode']=True
import numpy as np
import pandas as pd
import pylab
import scipy.cluster.hierarchy as sch
from matplotlib import font_manager
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import config, distance

# print(plt.style.available)
# plt.style.use('dark_background') #  'dark_background'

# from cogent.core.entity import unique

with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
        font_file = font_manager.findfont(font_manager.FontProperties(family='Arial'))
        matplotlib.rcParams["font.family"] = "Arial"
    except UserWarning:
        pass


def ncolors(n, colormap='jet'):
    """utility for defining N evenly spaced colors across a color map"""
    # colormap options: 'viridis' 'jet' 'gist_ncar', 'hsv'
    cmap = plt.get_cmap(colormap)
    cmap_max = cmap.N
    return [cmap(int(k * cmap_max / (n - 1))) for k in range(n)]


def dendrogram_plot(data_table, D=[], xlabels_order=[], xlabels=None, ylabels=[],
                    filename=config.output_dir + '/dendrogram', metric=config.similarity_method,
                    linkage_method="single",
                    colLable=False, rowLabel=False, color_bar=True, sortCol=True):
    # Adopted from Ref: http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
    plt.close()
    plt.close()
    matplotlib.rcParams['lines.linewidth'] = 0.25
    scale = []  # config.transform_method
    max_hight = 300
    max_weight = 300
    if not data_table is None:
        plot_height = min(int(len(data_table) / 7.25) + 5, max_hight)
        plot_weight = min(math.floor(len(data_table[0]) / len(data_table)) * plot_height,
                          min(int(len(data_table[0]) / 7.25) + 5, max_weight))
        # print plot_height, plot_weight
        if len(data_table) > 1000 or len(data_table[0]) > 1000:
            plot_dpi = 50
        else:
            plot_dpi = 300
        fig = pylab.figure(figsize=(plot_weight, plot_height), dpi=plot_dpi)
    else:
        plot_height = min(int(len(D) / 7.25) + 5, max_hight)
        plot_weight = plot_height
        fig = pylab.figure(figsize=(plot_weight, plot_height))
        #figsize = (cm2inch(4.5), cm2inch(4.5))
    fig = pylab.figure(figsize=(3, 2.5))
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6], frame_on=False)
    ax1.get_xaxis().set_tick_params(which='both', labelsize=8, top='off', bottom='off', direction='out')
    ax1.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', left='off', direction='out')
    # Compute and plot second dendrogram.
    if len(D) > 0:
        Y1 = linkage(squareform(D), method=linkage_method,  optimal_ordering=True)
    else:
        D = pdist(data_table, metric=distance.pDistance)
        Y1 = linkage(D, method=linkage_method, optimal_ordering = True)
    if len(Y1) > 1:
        try:
            Z1 = sch.dendrogram(Y1, orientation='left')
        except:
            print("Warning: dendrogram plot faced an exception!")
            pylab.close()
            Y1 = linkage(squareform(D), method=linkage_method, optimal_ordering = True)
            return Y1
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    if len(xlabels_order) == 0:
        ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2], frame_on=False)
        ax2.get_xaxis().set_tick_params(which='both', labelsize=8, top='off', bottom='off', direction='out')
        ax2.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', left='off', direction='out')
        Y2 = []
        if not data_table is None:
            try:
                Y2 = linkage(data_table.T, metric=distance.pDistance, method=linkage_method, optimal_ordering = True)
            except ValueError:
                pass
        if len(Y2) > 1:
            try:
                Z2 = sch.dendrogram(Y2)
            except:
                print("Warning: dendrogram 2 in hetamap plot faced an exception!")
                pylab.close()
                return Y1

        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.get_xaxis().set_tick_params(which='both', labelsize=8, top='off', bottom='off', direction='out')
        ax2.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', left='off', direction='out')
    else:
        Y2 = []

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    if len(Y1) > 1:
        idx1 = Z1['leaves']
    else:
        idx1 = [0]

    if len(Y2) > 1:
        idx2 = Z2['leaves']
    else:
        if len(D) > 0:
            idx2 = idx1
        else:
            idx2 = [0]
    if not data_table is None:
        data_table = data_table[idx1, :]
        if sortCol:
            if len(xlabels_order) == 0:
                data_table = data_table[:, idx2]
                xlabels_order.extend(idx2)
            else:
                data_table = data_table[:, xlabels_order]
    elif len(D) > 0:
        D = D.iloc[idx1, idx1]
    myColor = pylab.cm.YlOrBr
    if False:  # distance.c_hash_association_method_discretize[config.similarity_method]:
        myColor = pylab.cm.YlGnBu
    else:
        myColor = pylab.cm.RdBu_r
    if not data_table is None:
        scaled_values = data_table  # stats.scale_data(data_table, scale = scale)
    else:
        myColor = pylab.cm.pink
        scaled_values = D  # stats.scale_data(D, scale = scale)
    im = axmatrix.matshow(scaled_values, aspect='auto', origin='lower', cmap=myColor)  # YlGnBu
    if colLable:
        if len(ylabels) == len(idx2):
            label2 = [ylabels[i] for i in idx2]
        else:
            label2 = idx2
        # axmatrix.set_xticks(range(len(idx2)))
        # axmatrix.set_xticklabels(label2, minor=False)
        # axmatrix.xaxis.set_label_position('bottom')
        # axmatrix.xaxis.tick_bottom()
        axmatrix.get_xaxis().set_tick_params(which='both', labelsize=8, top='off', bottom='off', direction='out')
        axmatrix.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', left='off', direction='out')
    else:
        axmatrix.set_xticks([])
        axmatrix.set_xticklabels([])
        axmatrix.set_yticks([])
        axmatrix.set_yticklabels([])
        axmatrix.get_xaxis().set_tick_params(which='both', top='off')
        axmatrix.get_xaxis().set_tick_params(which='both', bottom='off')
        axmatrix.get_yaxis().set_tick_params(which='both', right='off')
        axmatrix.get_yaxis().set_tick_params(which='both', left='off')
        axmatrix.xaxis.set_label_position('bottom')
        axmatrix.xaxis.tick_bottom()

        # pylab.xticks(rotation=90, fontsize=6)
    if data_table:
        if rowLabel and len(data_table) / 7.25 < max_hight:
            if len(xlabels) == len(idx1):
                label1 = [xlabels[i] for i in idx1]
            else:
                label1 = idx1
        axmatrix.yaxis.set_label_position('right')
        axmatrix.set_yticklabels(label1, minor=False)
        axmatrix.set_yticks(range(len(idx1)))
        axmatrix.set_yticks([])
        axmatrix.set_yticklabels([])
        axmatrix.get_xaxis().set_tick_params(which='both', labelsize=8, top='off', bottom='off', direction='out')
        axmatrix.get_yaxis().set_tick_params(which='both', labelsize=8, right='off', left='off', direction='out')
        axmatrix.yaxis.tick_right()
        # pylab.yticks(rotation=0, fontsize=6)
    if color_bar:
        l = 0.2
        b = 0.71
        w = 0.02
        h = 0.2
        rect = l, b, w, h
        axcolor = fig.add_axes(rect)
        # axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
        legend_lable = " Distance"  # str(config.similarity_method).upper() if len(config.similarity_method) <5 else config.similarity_method.title()
        if len(scale) > 0:
            legend_lable = legend_lable + ' (' + str(scale.title()) + ')'
        fig.colorbar(im, cax=axcolor, label=legend_lable, ticks = [0, D.max().max()/2.0, D.max().max()])
        # pylab.colorbar(ax=axmatrix)
        # axmatrix.get_figure().colorbar(im, ax=axmatrix)
    # plt.tight_layout()

    fig.savefig(filename + '.pdf', bbox_inches='tight', dpi=350)
    # heatmap2(data_table, xlabels = xlabels, filename=filename+"_distance", metric = "nmi", method = "single", )
    # pylab.close()
    return Y1


def lda_ord(adist, X, y, cluster_members=None):
    pca = PCA(n_components=3)

    lda = LinearDiscriminantAnalysis(n_components=len(set(y)) - 1)
    X_r2 = lda.fit(X, y).transform(X)


def tsne_ord(adist, cluster_members=None, size_tobe_colored=3, metadata=None, shapeby=None, fig_size=[3, 2.5],
             point_size=None, show=False):
    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=0, metric='precomputed')
    coords = model.fit_transform(adist)
    ord_plot(coords, cluster_members=cluster_members, ord_name='t-SNE',
             size_tobe_colored=size_tobe_colored,
             xlabel='t-SNE 1', ylabel='t-SNE 2',
             metadata=metadata, shapeby=shapeby, fig_size=fig_size, point_size=point_size)
    model = TSNE(n_components=3, random_state=0, metric='precomputed')
    coords = model.fit_transform(adist)
    ord_plot_3d(coords, cluster_members=cluster_members, ord_name='t-SNE_3D',
             size_tobe_colored=size_tobe_colored,
             xlabel='t-SNE 1', ylabel='t-SNE 2', zlabel='t-SNE 3',
             metadata=metadata, shapeby=shapeby, fig_size=fig_size, point_size=point_size, show=show)


def mds_ord(adist, cluster_members=None, size_tobe_colored=3, metadata=None, shapeby=None, fig_size=[3, 2.5],
            point_size= None, show=False):

    #print(point_size, 50 / math.sqrt(adist.shape[0]))
    # pca = PCA(n_components=2)
    # X_r = pca.fit(X).transform(X)
    from sklearn import manifold

    adist = np.array(adist)
    amax = np.amax(adist)
    adist = adist / amax
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(adist)
    coords = results.embedding_
    ord_plot(coords, cluster_members=cluster_members, ord_name='MDS',
             size_tobe_colored=size_tobe_colored,
             xlabel='MDS stress (' + "{0:0.1f}".format(results.stress_) + ')',
             ylabel='', metadata=metadata, shapeby=shapeby, fig_size=fig_size, point_size=point_size)

    mds = manifold.MDS(n_components=3, dissimilarity="precomputed", random_state=6)
    results = mds.fit(adist)

    coords = results.embedding_
    ord_plot_3d(coords, cluster_members=cluster_members, ord_name='MDS_3D',
                size_tobe_colored=size_tobe_colored,
                xlabel='MDS stress (' + "{0:0.1f}".format(results.stress_) + ')',
                ylabel='', zlabel ='', metadata=metadata, shapeby=shapeby, fig_size=fig_size, point_size=point_size,
                show=show)


def pcoa_ord(X, cluster_members=None, size_tobe_colored=3, metadata=None, shapeby=None, fig_size=[3, 2.5],
             point_size=None, show=False):
    pca = PCA(n_components=3)
    pca_fit = pca.fit(X)
    X_r = pca_fit.transform(X)
    coords = X_r
    #from skbio import DistanceMatrix
    #from skbio.stats.ordination import pcoa

    #pcoa_results = pcoa(DistanceMatrix(X)).biplot_scores
    #coords = pcoa_results
    ord_plot(coords, cluster_members=cluster_members, ord_name='PCoA',
             size_tobe_colored=size_tobe_colored,
             xlabel='PCo1 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[0] * 100) + '%)',
             ylabel='PCo2 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[1] * 100) + '%)',
             metadata=metadata, shapeby=shapeby, fig_size=fig_size, point_size=point_size)
    ord_plot_3d(coords, cluster_members=cluster_members, ord_name='PCoA_3D',
             size_tobe_colored=size_tobe_colored,
             xlabel='PCo1 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[0] * 100) + '%)',
             ylabel='PCo2 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[1] * 100) + '%)',
             zlabel='PCo3 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[2] * 100) + '%)',
             metadata=metadata, shapeby=shapeby, fig_size=fig_size, point_size=point_size,
             show=show)


def pca_ord(X, cluster_members=None, size_tobe_colored=3, metadata=None, shapeby=None, fig_size=[3, 2.5],
            point_size=None, show=False):
    pca = PCA(n_components=2)
    pca_fit = pca.fit(X)
    X_r = pca_fit.transform(X)
    coords = X_r
    ord_plot(coords, cluster_members=cluster_members, ord_name='PCA',
             size_tobe_colored=size_tobe_colored,
             xlabel='PC1 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[0] * 100) + '%)',
             ylabel='PC2 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[1] * 100) + '%)',
             metadata=metadata, shapeby=shapeby, fig_size=fig_size, point_size=point_size)
    pca = PCA(n_components=3)
    pca_fit = pca.fit(X)
    X_r = pca_fit.transform(X)
    coords = X_r
    ord_plot_3d(coords, cluster_members=cluster_members, ord_name='PCA_3D',
                size_tobe_colored=size_tobe_colored,
                xlabel='PC1 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[0] * 100) + '%)',
                ylabel='PC2 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[1] * 100) + '%)',
                zlabel='PC3 (' + "{0:0.1f}".format(pca_fit.explained_variance_ratio_[2] * 100) + '%)',
                metadata=metadata, shapeby=shapeby, fig_size=fig_size, point_size=point_size,
                show=show)


def ord_plot(coords, cluster_members=None, ord_name='ord',
             size_tobe_colored=3, xlabel='First component',
             ylabel='Second component', metadata=None, shapeby=None, fig_size=[3, 2.5], point_size=10):

    # size of points if it not identified
    if point_size is None:
        point_size = min(10, 100 / math.sqrt(coords.shape[0]))

    # identify/filter outliers
    from scipy import stats
    '''if metadata is not None:
        metadata = metadata[(np.abs(stats.zscore(coords)) < 3).all(axis=1)]'''
    outliers = coords[(np.abs(stats.zscore(coords)) >= 3).any(axis=1)]

    #plt.close()
    plt.rcParams["figure.figsize"] = (fig_size[0], fig_size[1])
    ax = plt.axes()
    colors = ncolors(n=max(2, sum(
        [1 if len(cluster_members[target_name]) >= size_tobe_colored else 0 for target_name in cluster_members])))  #
    markers = ["o", "s", "v", "^", "D", "H", "d", "<", ">", "p",
               "P", "*", 'X', "h", "H", "+", "x", "1", "2", "3", "4", "8", ".", ",",
               "|"] + ["." for i in range(3000)]

    '''
     TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN, CARETLEFT,
               CARETRIGHT, CARETUP, CARETDOWN, CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE,
    '''
    order_metadata = []
    if metadata is not None and shapeby is not None:
        markers_dic = {'nan': "_"}
        df = metadata
        df['freq'] = df.groupby(shapeby)[shapeby].transform('count')
        order_metadata = list(df.sort_values(by=['freq'], ascending=False, na_position='last')[shapeby].unique())
        for i, val in enumerate(order_metadata):
            markers_dic[str(val)] = markers[i]
    cluster_members_large = {}
    cluster_members_small = {}
    sorted_key_by_len = sorted(cluster_members, key=lambda k: len(cluster_members[k]), reverse=True)
    for target_name in sorted_key_by_len:
        if len(cluster_members[str(target_name)]) >= size_tobe_colored:
            cluster_members_large[str(target_name)] = cluster_members[str(target_name)]
        else:
            cluster_members_small[str(target_name)] = cluster_members[str(target_name)]
    cmap = plt.get_cmap('jet')

    i = len(colors) - 1
    sorted_key_by_len_large = sorted(cluster_members_large, key=lambda k: len(cluster_members_large[k]), reverse=True)
    for target_name in sorted_key_by_len_large:
        label_flag = True
        if metadata is not None:
            point_markers = [markers_dic[str(val)] for val in
                             map(str, metadata[str(shapeby)].iloc[cluster_members_large[str(target_name)]])]
        else:
            point_markers = ['o' for val in cluster_members_large[target_name]]
        for xp, yp, mp in zip(coords[cluster_members_large[target_name], 0],
                              coords[cluster_members_large[target_name], 1],
                              point_markers):
            if [xp, yp] in outliers:
                # print [xp, yp]
                continue
            if label_flag:
                label_cluster = str(target_name) + ': ' + str(len(cluster_members_large[target_name]))
                label_flag = False
            else:
                label_cluster = None

            ax.scatter(xp,
                       yp,
                       color=colors[i],
                       marker=mp,
                       # label=label_cluster,
                       s=point_size, alpha=.8, linewidths=.05, edgecolors='black')

        i -= 1
    # Use legend with no shape for clusters as clusters
    # can have various values sof metadata value
    i = len(colors) - 1
    for target_name in sorted_key_by_len_large:
        label_cluster = str(target_name) + ': ' + str(len(cluster_members_large[target_name]))
        ax.scatter(None,
                   None,
                   color=colors[i],
                   marker='o',
                   label=label_cluster,
                   s=45, alpha=1, linewidths=.0, edgecolors=colors[i])
        i -= 1

    label_flag = True
    for target_name in cluster_members_small:
        if metadata is not None and shapeby is not None:
            point_mrakers = [markers_dic[val] for val in
                             map(str, metadata[str(shapeby)].iloc[cluster_members_small[target_name]])]
        else:
            point_mrakers = ['o' for val in cluster_members_small[target_name]]
        for xp, yp, mp in zip(coords[cluster_members_small[target_name], 0],
                              coords[cluster_members_small[target_name], 1],
                              point_mrakers):
            if [xp, yp] in outliers:
                # print [xp, yp]
                continue
            if label_flag:
                label_cluster = '#' + str(len(cluster_members_small)) + ' < ' + str(size_tobe_colored)
                label_flag = False
                ax.scatter(None,
                           None,
                           color='whitesmoke',
                           marker='o',
                           s=45, alpha=1, linewidths=0, edgecolors='black',
                           label=label_cluster)
            else:
                label_cluster = None
                ax.scatter(xp,
                           yp,
                           color='whitesmoke',
                           marker=mp,
                           s=point_size, alpha=.3, linewidths=.05, edgecolors='black',
                           label=label_cluster)

    # Legend and label for shape based on metadata
    ax.scatter(None,
               None,
               color='whitesmoke',
               marker=None,
               s=45, alpha=0, linewidths=.2, edgecolors='black',
               label=shapeby)

    metadata_leg_markers = [markers[i] for i in range(len(order_metadata))]
    # Get artists and labels for legend and chose which ones to display
    handles, labels = ax.get_legend_handles_labels()
    # Create custom artists
    markerArtist = [plt.Line2D([0], [0], color='black', marker=marker, \
                               linestyle="none", markeredgewidth=0.25, markerfacecolor="whitesmoke", ) \
                    for marker in metadata_leg_markers]

    to_display = max(1, len(handles))
    to_display_markers = min(10, len(markerArtist))
    ax.legend(
        [handle for i, handle in enumerate(handles) if i in range(to_display)] + markerArtist[0:to_display_markers],
        [label for i, label in enumerate(labels) if i in range(to_display)] + order_metadata[0:to_display_markers],
        loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster: size', shadow=False, scatterpoints=1,
        frameon=True, framealpha=.8, labelspacing=.6, fontsize=5)
    ax.get_legend().get_title().set_fontsize('6')
    ax.get_legend().get_title().set_weight('bold')

    ax.grid('off', axis='both')

    ax.set_xlabel(xlabel, fontsize=7, rotation=0, va='center', ha='center')
    ax.set_ylabel(ylabel, fontsize=7, rotation=90, va='center', ha='center')  # fontweight='bold',
    ax.get_xaxis().set_tick_params(which='both', labelsize=5, top='off', bottom='off', direction='out')
    ax.get_yaxis().set_tick_params(which='both', labelsize=5, right='off', left='off', direction='out')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['bottom'].set_linewidth(0.25)
    ax.spines['left'].set_linewidth(0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    try:
        ax.autoscale_view('tight')
    except:
        pass
    # ax.set_autoscale_on(False)
    # plt.title(ord_name + ' of omeClust clusters', fontsize=10)
    try:
        plt.tight_layout()
    except:
        pass
    if not shapeby is None:
        plt.savefig(config.output_dir + '/' + shapeby + '_' + ord_name + '_plot.pdf',
                    dpi=350, bbox_inches='tight')  # figsize=(2.0, 2.0) (cm2inch(8.9), cm2inch(8.9))
    else:
        plt.savefig(config.output_dir + '/' + ord_name + '_plot.pdf',
                    dpi=350, bbox_inches='tight')  # figsize=(2.0, 2.0) (cm2inch(8.9), cm2inch(8.9))
    plt.close()

def ord_plot_3d(coords, cluster_members=None, ord_name='ord', \
             size_tobe_colored=3, xlabel='First component',
             ylabel='Second component', zlabel='Third component',
             metadata=None, shapeby=None, fig_size=[3, 2.5], point_size=3, show = False):
    # size of points if it not identified
    if point_size is None:
        point_size = min(10, 50 / math.sqrt(coords.shape[0]))

    # identify/filter outliers

    '''if metadata is not None:
        metadata = metadata[(np.abs(stats.zscore(coords)) < 3).all(axis=1)]'''
    outliers = coords[(np.abs(stats.zscore(coords)) >= 3).any(axis=1)]

    plt.close()
    plt.rcParams["figure.figsize"] = (fig_size[0], fig_size[1])
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig) #, rect=[0, 0, 1, 1], elev=30, azim=-60)
    #ax =   plt.axes()
    colors = ncolors(n=max(2, sum(
        [1 if len(cluster_members[target_name]) >= size_tobe_colored else 0 for target_name in cluster_members])))  #
    markers = ["o", "s", "v", "^", "D", "H", "d", "<", ">", "p",
               "P", "*", 'X', "h", "H", "+", "x", "1", "2", "3", "4", "8", ".", ",",
               "|"] + ["." for i in range(3000)]

    '''
     TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN, CARETLEFT,
               CARETRIGHT, CARETUP, CARETDOWN, CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE,
    '''
    order_metadata = []
    if metadata is not None and shapeby is not None:
        markers_dic = {'nan': "_"}
        df = metadata
        df['freq'] = df.groupby(shapeby)[shapeby].transform('count')
        order_metadata = list(df.sort_values(by=['freq'], ascending=False, na_position='last')[shapeby].unique())
        for i, val in enumerate(order_metadata):
            markers_dic[str(val)] = markers[i]
    cluster_members_large = {}
    cluster_members_small = {}
    sorted_key_by_len = sorted(cluster_members, key=lambda k: len(cluster_members[k]), reverse=True)
    for target_name in sorted_key_by_len:
        if len(cluster_members[str(target_name)]) >= size_tobe_colored:
            cluster_members_large[str(target_name)] = cluster_members[str(target_name)]
        else:
            cluster_members_small[str(target_name)] = cluster_members[str(target_name)]
    #cmap = plt.get_cmap('jet')
    sorted_key_by_len_large = sorted(cluster_members_large, key=lambda k: len(cluster_members_large[k]), reverse=True)
    i = len(colors) - 1
    for target_name in sorted_key_by_len_large:
        label_cluster = str(target_name) + ': ' + str(len(cluster_members_large[target_name]))
        ax.scatter(None,
                   None,
                   0,
                   color=colors[i],
                   marker='o',
                   label=label_cluster,
                   depthshade=True,
                   s=45, alpha=1, linewidths=.0, edgecolors=colors[i])
        i -= 1

    label_flag = True
    for target_name in cluster_members_small:
        if metadata is not None and shapeby is not None:
            point_mrakers = [markers_dic[val] for val in
                             map(str, metadata[str(shapeby)].iloc[cluster_members_small[target_name]])]
        else:
            point_mrakers = ['o' for val in cluster_members_small[target_name]]
        for xp, yp, zp, mp in zip(coords[cluster_members_small[target_name], 0],
                                  coords[cluster_members_small[target_name], 1],
                                  coords[cluster_members_small[target_name], 2],
                                  point_mrakers):
            if [xp, yp, zp] in outliers:
                # print [xp, yp]
                continue
            if label_flag:
                label_cluster = '#' + str(len(cluster_members_small)) + ' < ' + str(size_tobe_colored)
                label_flag = False
                ax.scatter(None,
                           None,
                           0,
                           color='gainsboro',
                           marker='o',
                           depthshade=True,
                           s=45, alpha=1, linewidths=0, edgecolors='black',
                           label=label_cluster)
            else:
                label_cluster = None
                ax.scatter(xp,
                           yp,
                           zp,
                           color='gainsboro',
                           marker=mp,
                           depthshade=True,
                           s=point_size, alpha=.3, linewidths=0.05, edgecolors='black',
                           label=label_cluster)

    i = len(colors) - 1
    for target_name in sorted_key_by_len_large:
        label_flag = True
        if metadata is not None:
            point_markers = [markers_dic[str(val)] for val in
                             map(str, metadata[str(shapeby)].iloc[cluster_members_large[str(target_name)]])]
        else:
            point_markers = ['o' for val in cluster_members_large[target_name]]
        for xp, yp, zp, mp in zip(coords[cluster_members_large[target_name], 0],
                              coords[cluster_members_large[target_name], 1],
                              coords[cluster_members_large[target_name], 2],
                              point_markers):
            if [xp, yp, zp] in outliers:
                # print [xp, yp]
                continue
            if label_flag:
                label_cluster = str(target_name) + ': ' + str(len(cluster_members_large[target_name]))
                label_flag = False
            else:
                label_cluster = None

            ax.scatter(xp,
                       yp,
                       zp,
                       color=colors[i],
                       marker=mp,
                       # label=label_cluster,
                       depthshade=True,
                       s=point_size, alpha=.8, linewidths=0.05, edgecolors='black')

        i -= 1
    # Use legend with no shape for clusters as clusters
    # can have various values sof metadata value

    # Legend and label for shape based on metadata
    ax.scatter(None,
               None,
               0,
               color='gainsboro',
               marker=None,
               depthshade=True,
               s=45, alpha=0, linewidths=0, edgecolors='black',
               label=shapeby)

    metadata_leg_markers = [markers[i] for i in range(len(order_metadata))]
    # Get artists and labels for legend and chose which ones to display
    handles, labels = ax.get_legend_handles_labels()
    # Create custom artists
    markerArtist = [plt.Line2D([0], [0], color='black', marker=marker, \
                               linestyle="none", markeredgewidth=0.1, markerfacecolor="whitesmoke", ) \
                    for marker in metadata_leg_markers]

    to_display = max(1, len(handles))
    to_display_markers = min(10, len(markerArtist))
    ax.legend(
        [handle for i, handle in enumerate(handles) if i in range(to_display)] + markerArtist[0:to_display_markers],
        [label for i, label in enumerate(labels) if i in range(to_display)] + order_metadata[0:to_display_markers],
        loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster: size', shadow=False, scatterpoints=1,
        frameon=True, framealpha=.8, labelspacing=.6, fontsize=5)
    ax.get_legend().get_title().set_fontsize('6')
    ax.get_legend().get_title().set_weight('bold')

    ax.grid('on', axis='both')
    #ax.grid(True)
    xLabel =  ax.set_xlabel(xlabel, fontsize=7, rotation=0, va='center', ha='center')
    yLabel =  ax.set_ylabel(ylabel, fontsize=7, rotation=90, va='center', ha='center')  # fontweight='bold',
    zLabel = ax.set_zlabel(zlabel, fontsize=7, rotation=90, va='center', ha='center')  # fontweight='bold',
    ax.get_xaxis().set_tick_params(which='both', labelsize=5, top='off', bottom='off', direction='out')
    ax.get_yaxis().set_tick_params(which='both', labelsize=5, right='off', left='off', direction='out')
    ax.get_zaxis().set_tick_params(which='both', labelsize=5, right='off', left='off', direction='out')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])
    ax.xaxis.pane.set_linewidth(.1)
    ax.yaxis.pane.set_linewidth(.1)
    ax.zaxis.pane.set_linewidth(.1)
    ax.spines['bottom'].set_linewidth(0.1)
    ax.spines['left'].set_linewidth(0.1)
    ax.spines['right'].set_linewidth(0.1)
    ax.xaxis.labelpad = -18
    ax.yaxis.labelpad = -18
    ax.zaxis.labelpad = -18
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_linewidth(.25)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.xaxis.pane.fill = True
    #ax.yaxis.pane.fill = True
    #ax.zaxis.pane.fill = True

    # Now set color to white (or whatever is "invisible")
    #ax.xaxis.pane.set_edgecolor('w')
    #ax.yaxis.pane.set_edgecolor('w')
    #ax.zaxis.pane.set_edgecolor('w')
    warnings.filterwarnings("ignore")

    ax.grid(True)
    try:
        pass
        ax.autoscale_view('tight')
    except:
        pass
    ax.set_autoscale_on(True)
    # plt.title(ord_name + ' of omeClust clusters', fontsize=10)
    try:
        #pass
        plt.tight_layout()
        fig.tight_layout()
    except:
        pass
    #ax.view_init(30, 0)
    if show:
        plt.draw()
        plt.pause(100)
    #for angle in range(0, 360):
     #   ax.view_init(30, angle)
     #   plt.draw()
     #   plt.pause(.01)
    if not shapeby is None:
        plt.savefig(config.output_dir + '/' + shapeby + '_' + ord_name + '_plot.pdf',
                    dpi=350, bbox_inches='tight', figsize = (fig_size[0], fig_size[1]))  # figsize=(2.0, 2.0) (cm2inch(8.9), cm2inch(8.9))
    else:
        plt.savefig(config.output_dir + '/' + ord_name + '_plot.pdf',
                    dpi=350, bbox_inches='tight', figsize = (fig_size[0], fig_size[1]))  #  figsize = (fig_size[0], fig_size[1]) figsize=(2.0, 2.0) (cm2inch(8.9), cm2inch(8.9))
    plt.close()


def cm2inch(value):
    return value / 2.54


def pcoa(adist, cluster_members=None):
    from skbio import DistanceMatrix
    from skbio.stats.ordination import PCoA

    pcoa_results = PCoA(DistanceMatrix(adist)).scores()

    # print pcoa_results.


def parse_arguments():
    """ 
    Parse the arguments from the user
    """

    parser = argparse.ArgumentParser(
        description="omeClust visualization script.\n",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "adist",
        help="the input file D*N, Rows: D features and columns: N samples OR \n" +
             "a distance matrix file D*D (rows and columns should be the same and in the same order) \n ",
    )
    parser.add_argument(
        "clusters",
        help="the input file D*N, Rows: D features and columns: N samples OR \n" +
             "a distance matrix file D*D (rows and columns should be the same and in the same order) \n ",
    )
    parser.add_argument(
        "--metadata",
        help="metadata",
    )
    parser.add_argument(
        "--shapeby",
        type=str,
        help="the input file D*N, Rows: D features and columns: N samples OR \n" +
             "a distance matrix file D*D (rows and columns should be the same and in the same order) \n ",
    )
    parser.add_argument(
        "-o", "--output",
        help="the output directory\n",
        required=True)
    parser.add_argument(
        "--size-to-plot",
        type=int,
        dest='size_to_plot',
        default=3,
        help="Minimum size of cluster to be plotted")
    parser.add_argument("--fig-size", nargs=2,
                        # type=int,
                        dest='fig_size',
                        default=[3, 2.5], help="width and height of plots")
    parser.add_argument("--point-size",
                        type=int,
                        dest='point_size',
                        default=3, help="width and height of plots")
    parser.add_argument("--show",
                        help="show ordination plot before save\n",
                        action="store_true",
                        default=False,
                        dest='show')
    return parser.parse_args()


def main():
    args = parse_arguments()
    df_distance = pd.DataFrame()
    df_distance = pd.read_table(args.adist, index_col=0, header=0)
    df_distance = df_distance[df_distance.values.sum(axis=1) != 0]
    df_distance = df_distance[df_distance.values.sum(axis=0) != 0]
    # from scipy import stats
    # df_distance = df_distance[(np.abs(stats.zscore(df_distance)) < 3).all(axis=1)]
    # df_distance = df_distance.iloc[:, df_distance.index]
    metadata = None
    if args.shapeby and args.metadata:
        metadata = pd.read_table(args.metadata, index_col=0, header=0)
        metadata = metadata.loc[df_distance.index, :]
        # metadatadf.fillna(method='ffill')
    with open(args.clusters) as fin:
        next(fin)
        rows = (line.strip().split('\t') for line in fin)
        clusters = {row[0]: [list(df_distance).index(val) for val in row[1].split(';')] for row in rows}
    config.output_dir = args.output
    mds_ord(df_distance, cluster_members=clusters, size_tobe_colored=args.size_to_plot, metadata=metadata,
            shapeby=args.shapeby, fig_size=args.fig_size, point_size=args.point_size, show=args.show)
    pcoa_ord(df_distance, cluster_members=clusters, size_tobe_colored=args.size_to_plot, metadata=metadata,
             shapeby=args.shapeby, fig_size=args.fig_size, point_size=args.point_size, show=args.show)
    tsne_ord(df_distance, cluster_members=clusters, size_tobe_colored=args.size_to_plot, metadata=metadata,
             shapeby=args.shapeby, fig_size=args.fig_size, point_size=args.point_size, show=args.show)
    pca_ord(df_distance, cluster_members=clusters, size_tobe_colored=args.size_to_plot, metadata=metadata,
            shapeby=args.shapeby, fig_size=args.fig_size, point_size=args.point_size, show=args.show)
    # if data_flag:
    # pca_ord(df_data, cluster_members = dataprocess.cluster2dict(clusters, df_distance), size_tobe_colored = args.size_to_plot)
def network_plot(D, partition, min_weight = 0.5):
    #import community as community_louvain
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import networkx as nx

    # assume D is a distance matrix range between 0-max(D)
    W = D.max().max() - D

    # create edges from weight matrix
    W['from'] = list(W.index.values)
    W = pd.melt(W, id_vars=['from'], var_name='to', value_name='weight')
    W = W[W['weight'] >= min_weight]
    edges = [tuple(x) for x in W.to_records(index=False)]
    # initiate a graph
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # draw the graph
    pos = nx.spring_layout(G)
    #print(pos, partition)
    # color the nodes according to their partition
    clusters  = list(set(partition.values()))
    #print(clusters)
    cmap = ncolors(len(clusters)) # cm.get_cmap('jet', len(partition) ) #viridis
    colors_dic =  dict(zip(clusters,cmap))
    colors = []
    for i, val in enumerate(list(partition.values())):
        colors.append(colors_dic[val])
    #widths = [w *.1 for (a, b, w) in edges]
    weights = list(nx.get_edge_attributes(G, 'weight').values())
    weights = weights / max(weights) *.3
    #print (weights)

    #print (colors, len(colors))
    order_metadata = []
    #markers = ["o", "s", "v", "^", "D", "H", "d", "<", ">", "p",
    #           "P", "*", 'X', "h", "H", "+", "x", "1", "2", "3", "4", "8", ".", ",",
    #          "|", "_"] + ["." for i in range(3000)]
    #markers_dic = {'nan': "_"}
    #for i, val in enumerate(list(partition.values())):
    #    markers_dic[str(val)] = markers[i]
    #point_mrakers = [markers_dic[val] for val in
    #                 list(partition.values())]
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=5,
                           cmap=cmap,  node_color=colors, node_shape = 'o', edgecolors = 'black', linewidths =.05 )
    nx.draw_networkx_edges(G, pos, alpha=0.1, width = weights)
    #nx.set_xlabel(xlabel, fontsize=7, rotation=0, va='center', ha='center')
    #nx.get_xaxis().set_tick_params(which='both', labelsize=5, top='off', bottom='off', direction='out')
    #nx.get_yaxis().set_tick_params(which='both', labelsize=5, right='off', left='off', direction='out')
    #nx.get_xaxis().set_ticks([])
    #nx.get_yaxis().set_ticks([])
    plt.axis("off")
    plt.savefig(config.output_dir + '/' +'network_plot.pdf',
                dpi=350, bbox_inches='tight')  # figsize=(2.0, 2.0) (cm2inch(8.9), cm2inch(8.9))
    plt.close()

if __name__ == "__main__":
    main()
