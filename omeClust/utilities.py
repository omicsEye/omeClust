'''
This file includes modules that process finding clusters
===============================================
Author: Gholamali Rahnavard (gholamali.rahnavard@gmail.com)
'''
import math
import numpy as np
import pandas as pd
import sys
from itertools import product
from sklearn.metrics import normalized_mutual_info_score
from . import config
from . import omeClust


def predict_best_number_of_clusters(hierarchy_tree, distance_matrix):
    # distance_matrix = pd.DataFrame(distance_matrix)
    features = get_leaves(hierarchy_tree)
    clusters = []  # [hierarchy_tree]
    min_num_cluster = 2
    max_num_cluster = int(len(features) / math.ceil(math.log(len(features), 2)))
    best_sil_score = 0.0
    best_clust_size = 1
    for i in range(min_num_cluster, max_num_cluster):
        clusters = cutree_to_get_number_of_clusters(hierarchy_tree, distance_matrix, number_of_estimated_clusters=i)
        removed_singlton_clusters = [cluster for cluster in clusters if cluster.get_count() > 1]
        if len(removed_singlton_clusters) < 2:
            removed_singlton_clusters = clusters

        sil_scores = [sil for sil in silhouette_coefficient(removed_singlton_clusters, distance_matrix) if sil < 1.0]
        sil_score = np.mean(sil_scores)
        if best_sil_score < sil_score:
            best_sil_score = sil_score
            best_clust_size = len(clusters)
            result_sub_clusters = clusters

    print("The best guess for the number of clusters is: ", best_clust_size)
    return best_clust_size, clusters


def cutree_to_get_number_of_clusters(cluster, distance_matrix, number_of_estimated_clusters=None):
    n_features = cluster.get_count()
    if n_features == 1:
        return [cluster]
    if number_of_estimated_clusters is None:
        number_of_sub_cluters_threshold, _ = predict_best_number_of_clusters(cluster, distance_matrix)
        # round(math.log(n_features, 2))
    else:
        number_of_sub_cluters_threshold = number_of_estimated_clusters
    sub_clusters = []
    sub_clusters = chop_tree([cluster], level=0, skip=1)
    while len(sub_clusters) < number_of_sub_cluters_threshold:
        max_dist_node = sub_clusters[0]
        max_dist_node_index = 0
        for i in range(len(sub_clusters)):
            if max_dist_node.dist < sub_clusters[i].dist:
                max_dist_node = sub_clusters[i]
                max_dist_node_index = i
        if not max_dist_node.is_leaf():
            sub_clusters_to_add = chop_tree([max_dist_node], level=0, skip=1)
            del sub_clusters[max_dist_node_index]
            sub_clusters.insert(max_dist_node_index, sub_clusters_to_add[0])
            if len(sub_clusters_to_add) == 2:
                sub_clusters.insert(max_dist_node_index + 1, sub_clusters_to_add[1])
        else:
            break
    return sub_clusters


def chop_tree(cluster_nodes, level=0, skip=0):
    """
    This function chops the hierarchical tree from root, returning smaller tree towards the leaves 

    Parameters
    ---------------
        
        cluster_nodes : list of ClusterNode objects 
        level : int 
        skip : int 

    Output 
    ----------

        lC = list of ClusterNode objects 

    """
    iSkip = skip
    iLevel = level
    if iLevel < iSkip:
        try:
            return chop_tree(list(filter(lambda x: bool(x), [(p.right if p.right else None) for p in cluster_nodes])) \
                             + list(filter(lambda x: bool(x), [(q.left if q.left else None) for q in cluster_nodes]),
                                    level=iLevel + 1, skip=iSkip))
        except:
            return chop_tree([x for x in [(p.right if p.right else None) for p in cluster_nodes] if bool(x)] \
                             + [x for x in [(q.left if q.left else None) for q in cluster_nodes] if bool(x)],
                             level=iLevel + 1, skip=iSkip)

    elif iSkip == iLevel:
        if any(cluster_nodes):
            try:
                return list(filter(lambda x: bool(x), cluster_nodes))
            except:
                return [x for x in cluster_nodes if bool(x)]

        else:
            return []
            raise Exception("chop tree is malformed--empty!")


def get_leaves(cluster):
    return cluster.pre_order(lambda x: x.id)


def silhouette_coefficient(clusters, distance_matrix):
    # ====check within class homogeniety
    # Ref: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    # pMe = distance.c_hash_metric[config.Distance]
    distance_matrix = pd.DataFrame(distance_matrix)
    silhouette_scores = []
    if len(clusters) <= 1:
        sys.exit("silhouette method needs at least two clusters!")

    for i in range(len(clusters)):
        cluster_a = clusters[i].pre_order(lambda x: x.id)

        # find the next and previous clusters for each cluster as a
        # potential closest clusters to the cluster[i]
        if i == 0:
            next_cluster = prev_cluster = clusters[i + 1].pre_order(lambda x: x.id)
        elif i == len((clusters)) - 1:
            next_cluster = prev_cluster = clusters[i - 1].pre_order(lambda x: x.id)
        else:
            next_cluster = clusters[i + 1].pre_order(lambda x: x.id)
            prev_cluster = clusters[i - 1].pre_order(lambda x: x.id)

        s_all_a = []
        for a_feature in cluster_a:
            if len(cluster_a) == 1:
                a = 0.0
            else:
                temp_a_features = cluster_a[:]  # deepcopy(all_a_clusters)
                temp_a_features.remove(a_feature)
                a = np.mean([distance_matrix.iloc[i, j] for i, j in product([a_feature], temp_a_features)])
            b1 = np.mean([distance_matrix.iloc[i, j] for i, j in product([a_feature], next_cluster)])
            b2 = np.mean([distance_matrix.iloc[i, j] for i, j in product([a_feature], prev_cluster)])
            b = min(b1, b2)
            # print a, b
            s = (b - a) / max(a, b)
            s_all_a.append(s)
        silhouette_scores.append(np.mean(s_all_a))
    return silhouette_scores


def get_homogenous_clusters_silhouette(cluster, distance_matrix, number_of_estimated_clusters=None, resolution='high'):
    n = cluster.get_count()
    if n == 1:
        return [cluster]
    if False:  # resolution == 'low' :
        sub_clusters = cutree_to_get_number_of_clusters(cluster, distance_matrix,
                                                        number_of_estimated_clusters=number_of_estimated_clusters)
    else:
        sub_clusters = cutree_to_get_number_of_features(cluster, distance_matrix,
                                                        number_of_estimated_clusters=number_of_estimated_clusters)  # chop_tree([cluster], level=0, skip=1)#
    sub_silhouette_coefficient = silhouette_coefficient(sub_clusters, distance_matrix)
    while True:
        min_silhouette_node = sub_clusters[0]
        min_silhouette_node_index = 0

        # find cluster with minimum homogeneity 
        for i in range(len(sub_clusters)):
            if sub_silhouette_coefficient[min_silhouette_node_index] > sub_silhouette_coefficient[i]:
                min_silhouette_node = sub_clusters[i]
                min_silhouette_node_index = i
        # if the cluster with the minimum homogeneity has silhouette_coefficient
        # it means all cluster has passed the minimum homogeneity threshold  
        if sub_silhouette_coefficient[min_silhouette_node_index] == 1.0:
            break
        sub_clusters_to_check = cutree_to_get_number_of_features(min_silhouette_node, distance_matrix,
                                                                 number_of_estimated_clusters=number_of_estimated_clusters)  # chop_tree([min_silhouette_node], level=0, skip=1) #
        clusters_to_add = chop_tree([min_silhouette_node], level=0, skip=1)
        if len(clusters_to_add) < 2:
            break
        temp_silhouette_coefficient = silhouette_coefficient(clusters_to_add, distance_matrix)
        if len(sub_clusters_to_check) < 2:
            break
        sub_silhouette_coefficient_to_check = silhouette_coefficient(sub_clusters_to_check, distance_matrix)
        temp_sub_silhouette_coefficient_to_check = sub_silhouette_coefficient_to_check[:]
        temp_sub_silhouette_coefficient_to_check = [value for value in temp_sub_silhouette_coefficient_to_check if
                                                    value != 1.0]
        if resolution == 'low':
            if len(temp_sub_silhouette_coefficient_to_check) == 0 or sub_silhouette_coefficient[
                min_silhouette_node_index] >= min(temp_sub_silhouette_coefficient_to_check):
                sub_silhouette_coefficient[min_silhouette_node_index] = 1.0
            else:
                del sub_clusters[min_silhouette_node_index]  # min_silhouette_node)
                del sub_silhouette_coefficient[min_silhouette_node_index]
                sub_silhouette_coefficient.extend(temp_silhouette_coefficient)
                sub_clusters.extend(clusters_to_add)
        if resolution == 'high':
            if len(temp_sub_silhouette_coefficient_to_check) == 0 or sub_silhouette_coefficient[
                min_silhouette_node_index] >= max(temp_sub_silhouette_coefficient_to_check):
                sub_silhouette_coefficient[min_silhouette_node_index] = 1.0
            else:
                del sub_clusters[min_silhouette_node_index]  # min_silhouette_node)
                del sub_silhouette_coefficient[min_silhouette_node_index]
                sub_silhouette_coefficient.extend(temp_silhouette_coefficient)
                sub_clusters.extend(clusters_to_add)
        if resolution == 'medium':
            if len(temp_sub_silhouette_coefficient_to_check) == 0 or sub_silhouette_coefficient[
                min_silhouette_node_index] >= np.mean(temp_sub_silhouette_coefficient_to_check):
                sub_silhouette_coefficient[min_silhouette_node_index] = 1.0
            else:
                del sub_clusters[min_silhouette_node_index]  # min_silhouette_node)
                del sub_silhouette_coefficient[min_silhouette_node_index]
                sub_silhouette_coefficient.extend(temp_silhouette_coefficient)
                sub_clusters.extend(clusters_to_add)

    return sub_clusters


def cutree_to_get_number_of_features(cluster, distance_matrix, number_of_estimated_clusters=None):
    n_features = cluster.get_count()
    if n_features == 1:
        return [cluster]
    if number_of_estimated_clusters is None:
        number_of_estimated_clusters = math.sqrt(n_features)  # math.log(n_features, 2)
    # sub_clusters = []
    sub_clusters = chop_tree([cluster], level=0, skip=1)
    while True:  # not all(val <= t for val in distances):
        # print(sub_clusters)
        largest_node = sub_clusters[0]
        index = 0
        for i in range(len(sub_clusters)):
            if largest_node.get_count() < sub_clusters[i].get_count():
                largest_node = sub_clusters[i]
                index = i
        if largest_node.get_count() > (n_features / number_of_estimated_clusters):
            # print(largest_node.get_count(), n_features, number_of_estimated_clusters)
            # sub_clusters.remove(largest_node)
            # sub_clusters = sub_clusters[:index] + sub_clusters[index+1 :]
            del sub_clusters[index]
            sub_clusters += chop_tree([largest_node], level=0, skip=1)
        else:
            break
    return sub_clusters


def omeClust_enrichment_score(clusters, metadata, method = "NMI"):
    metadata_enrichment_score = dict()
    # sorted_keys = []
    # metadata_enrichment_score['resolution_score'] = weighted_hormonic_mean(clusters, [], n)
    if metadata is not None:
        for meta in metadata.columns:
            if len(metadata[meta].unique()) < 2:
                continue
            #metadata[meta] = jenks_discretize(metadata[meta], number_of_bins=None)
            #print(metadata[meta])
            # based of unique value decide if it  need decritziation
            #if len(set(metadata[meta])) > round(math.sqrt(len(metadata[meta]))):
            try:
                #print(list(metadata[meta]))
                metadata[meta] = omeClust_discretize(metadata[meta])#jenks_discretize(metadata[meta], number_of_bins=None)#
                #print(list(metadata[meta]))
            except:
                pass
            meta_enrichment_score = []

            i= 0
            membership = []
            all_cluster_members = []
            if metadata is not None:

                for cluster in clusters:
                    i += 1
                    cluster_members = cluster.pre_order(lambda x: x.id)
                    all_cluster_members.extend(cluster_members)
                    membership.extend(['C'+str(i) for member in cluster_members])
                    # get category with max frequency in the cluster for meta column as metadata
                    if method.lower() == "freq":
                        freq_metadata, freq_value = most_common(metadata[meta].iloc[cluster_members])
                        if freq_metadata != '':
                            meta_enrichment_score.append(freq_value / len(cluster_members))
                        else:
                            meta_enrichment_score.append('')
                #print(membership)
                    # calculate the cluster score
                # calculate meta data score for metadata meta
                metadata_enrichment_score[meta] = meta_enrichment_score
                if method.lower() == "nmi":
                    # calculate NMI as an enrichment score
                    new_X, new_Y = remove_pairs_with_a_missing(membership, list(metadata[meta].iloc[all_cluster_members]))
                    nmi_score = normalized_mutual_info_score(new_X, new_Y)
                    metadata_enrichment_score[meta] = [nmi_score for cluster in clusters]
            # weighted_hormonic_mean(clusters, meta_enrichment_score, n)
            # print metadata_enrichment_score[meta]
        # sorted_keys = sorted(metadata_enrichment_score, key=lambda k: sum(metadata_enrichment_score[k]), reverse=True)
        metadata.to_csv(config.output_dir + '/discretize_metadata.txt', sep='\t')
    # sorted_keys = sorted(metadata_enrichment_score, key=lambda k: sum(metadata_enrichment_score[k]), reverse=True)

    metadata_enrichment_score['resolution_score'] = resolution_score(clusters)
    metadata_enrichment_score['n'] = [clusters[i].count for i in range(len(clusters))]
    metadata_enrichment_score['branch_condensed_distance'] = [clusters[i].dist for i in range(len(clusters))]

    metadata_enrichment_score_df = pd.DataFrame.from_dict(metadata_enrichment_score)
    #print(metadata_enrichment_score_df)
    metadata_enrichment_score_df2 = metadata_enrichment_score_df[metadata_enrichment_score_df['resolution_score'] > 0.05]
    # if there is ate least on cluster passes threshold use it as major cluster
    #print(metadata_enrichment_score_df2.shape)
    if metadata_enrichment_score_df2.shape[0] >1:
        metadata_enrichment_score_df = metadata_enrichment_score_df2
    #print(metadata_enrichment_score_df)
    # print(metadata_enrichment_score_df.mean())
    # print("index before",metadata_enrichment_score_df.columns)
    sorted_keys = list(metadata_enrichment_score_df.mean().sort_values(ascending=False).index)
    # metadata_enrichment_score_df.reindex(list(metadata_enrichment_score_df.mean().sort_values(ascending=False).index), axis=1)
    # print("index after", metadata_enrichment_score_df.columns)
    # print("after reindex:",metadata_enrichment_score_df)
    #print(metadata_enrichment_score_df['n'], metadata_enrichment_score['n'])
    config.size_to_plot = min(metadata_enrichment_score_df['n'])
    print("The number of major clusters: ", metadata_enrichment_score_df.shape[0])
    # metadata_enrichment_score_df = metadata_enrichment_score_df.reindex(
    #    columns=(['resolution_score'] + list([a for a in metadata_enrichment_score_df.columns
    #                                                                                   if a != 'resolution_score'])))
    # metadata_enrichment_score_df = metadata_enrichment_score_df.reindex(
    #    columns=(['n'] + list([a for a in metadata_enrichment_score_df.columns if a != 'n'])))

    # sorted_keys = list(metadata_enrichment_score_df.columns)
    # print(sorted_keys)
    sorted_keys.remove('branch_condensed_distance')
    sorted_keys.insert(0, 'branch_condensed_distance')
    sorted_keys.remove('resolution_score')
    sorted_keys.insert(0, 'resolution_score')
    sorted_keys.remove('n')
    sorted_keys.insert(0, 'n')

    return metadata_enrichment_score, sorted_keys


def most_common(lst):
    from collections import Counter
    #print("before", lst)
    lst = [x for x in lst if str(x) != 'nan']
    #print ("After",lst)
    data = Counter(lst)
    if len(lst) == 0:
        return '', ''
    freq_metadata = max(lst, key=data.get)
    freq_value = data[freq_metadata]
    return freq_metadata, freq_value


def resolution_score(clusters):
    n = sum([clusters[i].count for i in range(len(clusters))])
    scores = [1.0 / (.5 / (clusters[i].count / n) + .5 / (1.0 - clusters[i].dist)) for i in
              range(len(clusters))]
    return scores

def louvain_clust(D, min_weight = 0.5):
    import community as community_louvain
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import networkx as nx

    # assume D is a distance matrix range between 0-1
    W = 1.0 - D

    # create edges from weight matrix
    W['from'] = list(W.index.values)
    W = pd.melt(W, id_vars=['from'], var_name='to', value_name='weight')
    W = W[W['weight'] >= min_weight]
    edges = [tuple(x) for x in W.to_records(index=False)]
    # initiate a graph
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # compute the best partition
    partition = community_louvain.best_partition(G)
    # draw the graph
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('jet', max(partition.values()) + 1) #viridis
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
                           cmap=cmap, node_color=list(partition.values()), node_shape = 'o' )
    nx.draw_networkx_edges(G, pos, alpha=0.01) #, width = 2.0)
    #nx.set_xlabel(xlabel, fontsize=7, rotation=0, va='center', ha='center')
    #nx.get_xaxis().set_tick_params(which='both', labelsize=5, top='off', bottom='off', direction='out')
    #nx.get_yaxis().set_tick_params(which='both', labelsize=5, right='off', left='off', direction='out')
    #nx.get_xaxis().set_ticks([])
    #nx.get_yaxis().set_ticks([])

    plt.savefig(config.output_dir + '/' +'louvain_clust_plot.pdf',
                dpi=350)  # figsize=(2.0, 2.0) (cm2inch(8.9), cm2inch(8.9))
    plt.close()


def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value <= breaks[i]:
            return "G " + str(i)
    return "G " + str(len(breaks) - 1)


def jenks_discretize(values, number_of_bins=None):
    import jenkspy
    import math
    if number_of_bins is None:
        number_of_bins = min(len(set(values)), round(math.sqrt(len(values))))
    #print(number_of_bins)
    breaks = jenkspy.jenks_breaks(values, int(number_of_bins))
    values_in_bins = [classify(value, breaks) for value in values]
    return values_in_bins

def omeClust_discretize(values, linkage_method='complete', resolution='low' ):
    """
    Discretize values
    :param values: a list of numeric values
    :return: bins membership reflecting category of each value
    """
    df_distance = abs(np.array([values], dtype=float).T - np.array([values], dtype=float))
    df_distance = pd.DataFrame(df_distance)
    #print(df_distance)
    cluster_bins = omeClust.main_run(distance_matrix=df_distance,
                                     number_of_estimated_clusters=2,
                                     linkage_method=linkage_method,
                                     output_dir=config.output_dir, do_plot=False, resolution=resolution)
    membership = []
    all_bin_members = []
    i=0
    membership = ["G 0" for i in range(len(values))]
    for cluster in cluster_bins:
        i += 1
        cluster_members = cluster.pre_order(lambda x: x.id)
        all_bin_members.extend(cluster_members)
        #print(cluster_members)
        for l in cluster_members:
            membership[l] = 'G ' + str(i) #.extend(['G ' + str(i) for member in cluster_members])
        #print (i, ":", len(cluster_members))
    return membership #membership[all_bin_members]

def remove_pairs_with_a_missing(X, Y, missing_char='NaN'):
    if missing_char != missing_char or missing_char == 'NaN' :
        test = [missing_char in [a, b] or not a == a or not b == b for a, b in zip(X, Y)]
    else:
        test = [missing_char in [a, b] for a, b in zip(X, Y)]
    new_X = [a for a, b in zip(X, test) if not b]
    new_Y = [a for a, b in zip(Y, test) if not b]
    # print new_Y
    return (new_X, new_Y)
