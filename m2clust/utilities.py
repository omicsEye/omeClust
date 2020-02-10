
'''
This file includes modules that process finding clusters
===============================================
Author: Gholamali Rahnavard (gholamali.rahnavard@gmail.com)
'''
import math
import sys
import pandas
import numpy as np
from itertools import product, combinations

def predict_best_number_of_clusters(hierarchy_tree, distance_matrix):
    #distance_matrix = pandas.DataFrame(distance_matrix)
    features = get_leaves(hierarchy_tree)
    clusters= [] #[hierarchy_tree]
    min_num_cluster = 2  
    max_num_cluster = int(len(features)/math.ceil(math.log(len(features), 2)))
    best_sil_score = 0.0
    best_clust_size = 1
    for i in range(min_num_cluster,max_num_cluster):
        clusters = cutree_to_get_number_of_clusters(hierarchy_tree, distance_matrix, number_of_estimated_clusters=i)
        removed_singlton_clusters = [cluster for cluster in clusters if cluster.get_count()>1]
        if len(removed_singlton_clusters) < 2:
            removed_singlton_clusters = clusters

        sil_scores = [sil for sil in silhouette_coefficient(removed_singlton_clusters, distance_matrix) if sil < 1.0]
        sil_score = np.mean(sil_scores)
        if best_sil_score < sil_score:
            best_sil_score = sil_score
            best_clust_size = len(clusters)
            result_sub_clusters = clusters
                
    print ("The best guess for the number of clusters is: ", best_clust_size)
    return best_clust_size, clusters   



def cutree_to_get_number_of_clusters (cluster, distance_matrix, number_of_estimated_clusters = None):
    n_features = cluster.get_count()
    if n_features==1:
        return [cluster]
    if number_of_estimated_clusters is None:
        number_of_sub_cluters_threshold, _ = predict_best_number_of_clusters(cluster, distance_matrix)
        #round(math.log(n_features, 2))        
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
            sub_clusters.insert(max_dist_node_index,sub_clusters_to_add[0])
            if len(sub_clusters_to_add) ==2:
                sub_clusters.insert(max_dist_node_index+1,sub_clusters_to_add[1])
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
            + list(filter(lambda x: bool(x), [(q.left if q.left else None) for q in cluster_nodes]), level=iLevel + 1, skip=iSkip))
        except:
            return chop_tree([x for x in [(p.right if p.right else None) for p in cluster_nodes] if bool(x)] \
            + [x for x in [(q.left if q.left else None) for q in cluster_nodes] if bool(x)], level=iLevel + 1, skip=iSkip) 

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
    #====check within class homogeniety
    #Ref: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    #pMe = distance.c_hash_metric[config.Distance]
    distance_matrix = pandas.DataFrame(distance_matrix)
    silhouette_scores = []
    if len(clusters) <= 1:
        sys.exit("silhouette method needs at least two clusters!")
        
    for i in range(len(clusters)):
        cluster_a = clusters[i].pre_order(lambda x: x.id)
        
        # find the next and previous clusters for each cluster as a
        # potential closest clusters to the cluster[i]
        if i==0:
            next_cluster = prev_cluster = clusters[i+1].pre_order(lambda x: x.id)
        elif i == len((clusters))-1:
            next_cluster = prev_cluster = clusters[i-1].pre_order(lambda x: x.id) 
        else:
            next_cluster = clusters[i+1].pre_order(lambda x: x.id)
            prev_cluster = clusters[i-1].pre_order(lambda x: x.id)

        s_all_a = []
        for a_feature in cluster_a:
            if len(cluster_a) ==1:
                a = 0.0
            else:
                temp_a_features = cluster_a[:]#deepcopy(all_a_clusters)
                temp_a_features.remove(a_feature)
                a = np.mean([distance_matrix.iloc[i, j] for i,j in product([a_feature], temp_a_features)])            
            b1 = np.mean([distance_matrix.iloc[i, j] for i,j in product([a_feature], next_cluster)])
            b2 = np.mean([distance_matrix.iloc[i, j] for i,j in product([a_feature], prev_cluster)])
            b = min(b1,b2)
            #print a, b
            s = (b-a)/max(a,b)
            s_all_a.append(s)
        silhouette_scores.append(np.mean(s_all_a))
    return silhouette_scores

def get_homogenous_clusters_silhouette(cluster, distance_matrix, number_of_estimated_clusters= None, resolution= 'high'):
    n = cluster.get_count()
    if n==1:
        return [cluster]
    if False:#resolution == 'low' :
        sub_clusters = cutree_to_get_number_of_clusters(cluster, distance_matrix, number_of_estimated_clusters= number_of_estimated_clusters)    
    else:
        sub_clusters = cutree_to_get_number_of_features(cluster, distance_matrix, number_of_estimated_clusters= number_of_estimated_clusters)#chop_tree([cluster], level=0, skip=1)#
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
        sub_clusters_to_check = cutree_to_get_number_of_features(min_silhouette_node, distance_matrix, number_of_estimated_clusters=number_of_estimated_clusters) #chop_tree([min_silhouette_node], level=0, skip=1) #
        clusters_to_add = chop_tree([min_silhouette_node], level=0, skip=1)
        if len(clusters_to_add) < 2:
            break
        temp_silhouette_coefficient = silhouette_coefficient(clusters_to_add, distance_matrix)
        if len(sub_clusters_to_check) < 2:
            break
        sub_silhouette_coefficient_to_check = silhouette_coefficient(sub_clusters_to_check, distance_matrix)
        temp_sub_silhouette_coefficient_to_check = sub_silhouette_coefficient_to_check[:]
        temp_sub_silhouette_coefficient_to_check = [value for value in temp_sub_silhouette_coefficient_to_check if value != 1.0]
        if resolution == 'low': 
            if len(temp_sub_silhouette_coefficient_to_check) == 0 or sub_silhouette_coefficient[min_silhouette_node_index] >= min(temp_sub_silhouette_coefficient_to_check):
                sub_silhouette_coefficient[min_silhouette_node_index] =  1.0
            else:
                del sub_clusters[min_silhouette_node_index]#min_silhouette_node)
                del sub_silhouette_coefficient[min_silhouette_node_index]
                sub_silhouette_coefficient.extend(temp_silhouette_coefficient)
                sub_clusters.extend(clusters_to_add)
        if resolution == 'high':
            if len(temp_sub_silhouette_coefficient_to_check) == 0 or sub_silhouette_coefficient[min_silhouette_node_index] >= max(temp_sub_silhouette_coefficient_to_check):
                sub_silhouette_coefficient[min_silhouette_node_index] =  1.0
            else:
                del sub_clusters[min_silhouette_node_index]#min_silhouette_node)
                del sub_silhouette_coefficient[min_silhouette_node_index]
                sub_silhouette_coefficient.extend(temp_silhouette_coefficient)
                sub_clusters.extend(clusters_to_add)
        if resolution == 'medium':
            if len(temp_sub_silhouette_coefficient_to_check) == 0 or sub_silhouette_coefficient[min_silhouette_node_index] >= np.mean(temp_sub_silhouette_coefficient_to_check):
                sub_silhouette_coefficient[min_silhouette_node_index] =  1.0
            else:
                del sub_clusters[min_silhouette_node_index]#min_silhouette_node)
                del sub_silhouette_coefficient[min_silhouette_node_index]
                sub_silhouette_coefficient.extend(temp_silhouette_coefficient)
                sub_clusters.extend(clusters_to_add)
   
    return sub_clusters

def cutree_to_get_number_of_features (cluster, distance_matrix, number_of_estimated_clusters=None):
    n_features = cluster.get_count()
    if n_features == 1:
        return [cluster]
    if number_of_estimated_clusters is None:
        number_of_estimated_clusters = math.sqrt(n_features)#math.log(n_features, 2)
    #sub_clusters = []
    sub_clusters = chop_tree([cluster], level=0, skip=1)
    while True:# not all(val <= t for val in distances):
        #print(sub_clusters)
        largest_node = sub_clusters[0]
        index = 0
        for i in range(len(sub_clusters)):
            if largest_node.get_count() < sub_clusters[i].get_count():
                largest_node = sub_clusters[i]
                index = i
        if largest_node.get_count() > (n_features/number_of_estimated_clusters):
            #print(largest_node.get_count(), n_features, number_of_estimated_clusters)
            #sub_clusters.remove(largest_node)
            #sub_clusters = sub_clusters[:index] + sub_clusters[index+1 :]
            del sub_clusters[index]
            sub_clusters += chop_tree([largest_node], level=0, skip=1)
        else:
            break
    return sub_clusters
def m2clust_score(clusters, metadata, n):
    rahnavard_metadata_score = dict()
    sorted_keys = []
    #rahnavard_metadata_score['resolution_score'] = weighted_hormonic_mean(clusters, [], n)
    if metadata is not None:
        for meta in metadata.columns:
            if len(metadata[meta].unique())<2:
                continue 
            rahnavard_meta_score = []
            if metadata is not None:
                for cluster in clusters:
                    cluster_members = cluster.pre_order(lambda x: x.id)
                    # get category with max frequency in the cluster for meta column as metadata
                    freq_metadata, freq_value =  most_common(metadata.ix[cluster_members, meta])
                    rahnavard_meta_score.append( freq_value*1.0/len(cluster_members) )
                    # calculate the cluster score
                # calculate meta data score for metadata meta
            rahnavard_metadata_score[meta] = weighted_hormonic_mean(clusters, rahnavard_meta_score, n)
            #print rahnavard_metadata_score[meta] 
        sorted_keys = sorted(rahnavard_metadata_score, key=lambda k: sum(rahnavard_metadata_score[k]), reverse=True)
    rahnavard_metadata_score['resolution_score'] = weighted_hormonic_mean(clusters, [], n)
    sorted_keys = sorted(rahnavard_metadata_score, key=lambda k: sum(rahnavard_metadata_score[k]), reverse=True)
    #sorted_keys += ['resolution_score']
    return rahnavard_metadata_score, sorted_keys
def most_common(lst):
    from collections import Counter
    data = Counter(lst)
    freq_metadata = max(lst, key=data.get)
    freq_value =  data[freq_metadata]
    return freq_metadata, freq_value
def weighted_hormonic_mean(clusters, rahnavard_meta_score = None, n = 0):
    w = [2.0, 1.0, 1.0]
    #print rahnavard_meta_score
    if len(rahnavard_meta_score) > 0 :
        w_h_mean = [sum(w)/ ( w[0]/(clusters[i].count*1.0/n) + w[1]/rahnavard_meta_score[i] + w[2]/(1.0- clusters[i].dist) ) for i in range(len(clusters))]
    else: 
        w_h_mean = [sum(w)/ ( w[0]/(clusters[i].count*1.0/n) +  w[2]/(1.0- clusters[i].dist) ) for i in range(len(clusters))]
    return w_h_mean

