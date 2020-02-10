#!/usr/bin/env python

"""
Multi-resolution clustering approach to find clusters with different diameter 

"""
import argparse
import sys
import tempfile
import os
import shutil
import re
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import to_tree, linkage
from scipy.spatial.distance import pdist, squareform
import numpy as np
from numpy import array
try:
    from . import utilities, config, dataprocess
except ImportError:
    sys.exit("CRITICAL ERROR: Unable to find the utilities module." + 
        " Please check your m2clust install.")
from . import distance, viz
def main_run(distance_matrix=None,
              number_of_estimated_clusters =None,
              linkage_method = 'single', 
              output_dir=None, 
              do_plot=True,
              resolution='low'):
    bTree=True

    if do_plot:
        Z = viz.dendrogram_plot(data_table=None , D=distance_matrix, xlabels_order = [], xlabels=distance_matrix.index,
                     filename=output_dir+"/m2clust_dendrogram", colLable=False, linkage_method=linkage_method)
    else:
        Z = linkage(distance_matrix, method= linkage_method)
    
    hclust_tree = to_tree(Z) 
    #clusters = cutree_to_get_below_threshold_number_of_features (hclust_tree, t = estimated_num_clust)
    if number_of_estimated_clusters == None:
        number_of_estimated_clusters,_ = utilities.predict_best_number_of_clusters(hclust_tree, distance_matrix)
    clusters = utilities.get_homogenous_clusters_silhouette(hclust_tree, array(distance_matrix),
                                                            number_of_estimated_clusters=number_of_estimated_clusters,
                                                            resolution=resolution)
    #print [cluster.pre_order(lambda x: x.id) for cluster in clusters]
    #print(len(clusters))
    return clusters

def check_requirements():
    """
    Check requirements (file format, dependencies, permissions)
    """
    # check the third party softwares for plotting the results
    try:
        import pandas as pd
    except ImportError:
        sys.exit("--- Please check your installation for pandas library")
    # Check that the output directory is writeable
    output_dir = os.path.abspath(config.output_dir)
    if not os.path.isdir(output_dir):
        try:
            print("Creating output directory: " + output_dir)
            os.mkdir(output_dir)
        except EnvironmentError:
            sys.exit("CRITICAL ERROR: Unable to create output directory.")
def parse_arguments(args):
    """ 
    Parse the arguments from the user
    """
    
    parser = argparse.ArgumentParser(
        description= "Multi-resolution clustering using hierarchical clustering and Silhouette score.\n",
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "-i", "--input",
        help="the input file D*N, Rows: D features and columns: N samples OR \n"+
        "a distance matrix file D*D (rows and columns should be the same and in the same order) \n ",
        required=False)
    parser.add_argument(
        "-o", "--output",
        help="the output directory\n",
        required=True)
    parser.add_argument(
        "-m", "--similarity",
        default='spearman',
        help="similarity measurement {default spearman, options: spearman, nmi, ami, dmic, mic, pearson, dcor}")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Rows are features and each column is a metadata")
    parser.add_argument(
        "-n", "--estimated_number_of_clusters",
        type=int,
        default=2,
        help="estimated number of clusters")
    parser.add_argument(
        "--size-to-plot",
        type=int,
        dest='size_to_plot',
        default=3,
        help="Minimum size of cluster to be plotted")
    parser.add_argument(
        "-c", "--linkage_method",
        default='single',
        help="linkage clustering method method {default = single, options average, complete\n")
    parser.add_argument(
        "--plot", 
        help="dendrogram plus heatmap\n", 
        action="store_true",
        default=False)
    parser.add_argument(
        "--resolution", 
        default='low',
        help="Resolution c .\
         Low resolution is good when clusters are well separated clusters.",
        choices=['high', 'medium', 'low'])
    parser.add_argument(
        "-v", "--verbose",
        help="additional output is printed\n", 
        action="store_true",
        default=False)
    return parser.parse_args()


def m2clust(data, metadata, resolution=config.resolution,
           output_dir=config.output_dir,
           estimated_number_of_clusters=config.estimated_number_of_clusters,
           linkage_method=config.linkage_method, plot=config.plot, size_to_plot=config.size_to_plot):
    config.output_dir = output_dir
    check_requirements()
    data_flag = True

    if all(a == b for (a, b) in zip(data.columns, data.index)):
        df_distance = data
        data_flag = False
    else:
        df_distance = pd.DataFrame(squareform(pdist(data, metric=distance.pDistance)), index=data.index,
                                   columns=data.index)
    df_distance = df_distance[df_distance.values.sum(axis=1) != 0]
    df_distance = df_distance[df_distance.values.sum(axis=0) != 0]
    df_distance.to_csv(output_dir + '/adist.txt', sep='\t')
    # df_distance = stats.scale_data(df_distance, scale = 'log')

    # viz.tsne_ord(df_distance, target_names = data.columns)
    clusters = main_run(distance_matrix=df_distance,
                        number_of_estimated_clusters=estimated_number_of_clusters,
                        linkage_method=linkage_method,
                        output_dir=output_dir, do_plot=plot, resolution=resolution)
    m2clust_scores, sorted_keys = None, None
    shapeby = None
    if metadata is not None:
        m2clust_scores, sorted_keys = utilities.m2clust_score(clusters, metadata, len(metadata))
        if len(sorted_keys) > 1:
            shapeby = sorted_keys[1]
    else:
        m2clust_scores, sorted_keys = utilities.m2clust_score(clusters, metadata, df_distance.shape[0])
    # print m2clust_scores, sorted_keys
    dataprocess.write_output(clusters, output_dir, df_distance, m2clust_scores, sorted_keys)

    viz.mds_ord(df_distance, target_names=dataprocess.cluster2dict(clusters, df_distance), \
                size_tobe_colered=size_to_plot, metadata=metadata, shapeby=shapeby)
    viz.pcoa_ord(df_distance, target_names=dataprocess.cluster2dict(clusters, df_distance), \
                 size_tobe_colered=size_to_plot, metadata=metadata, shapeby=shapeby)
    viz.tsne_ord(df_distance, target_names=dataprocess.cluster2dict(clusters, df_distance), \
                 size_tobe_colered=size_to_plot, metadata=metadata, shapeby=shapeby)
    if data_flag:
        viz.pca_ord(data, target_names=dataprocess.cluster2dict(clusters, df_distance), \
                    size_tobe_colered=size_to_plot, metadata=metadata, shapeby=shapeby)


def main( ):
    # Parse arguments from command line
    args = parse_arguments(sys.argv)
    config.similarity = args.similarity
    config.output_dir = args.output+"/"
    
    dataprocess.create_output(config.output_dir)
    config.input = args.input
    config.output = args.output
    config.resolution = args.resolution
    config.out_dir = args.output
    config.estimated_number_of_clusters = args.estimated_number_of_clusters
    config.linkage_method = args.linkage_method
    config.plot = args.plot
    config.size_to_plot = config.size_to_plot
    input_df = pd.DataFrame()
    df_data = pd.DataFrame()
    df_data = pd.read_table(config.input, index_col=0, header=0)

    config.metadata = args.metadata
    if config.metadata is not None:
        config.metadata = pd.read_table(config.metadata, index_col=0, header=0)
        config.metadata = config.metadata.loc[df_data.index, :]
    m2clust(data=df_data, metadata=config.metadata,
           resolution=config.resolution, output_dir=config.output)
if __name__ == "__main__":
    main( )