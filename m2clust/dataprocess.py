
import os
import sys
import shutil
import numpy

def create_output(output):
    # write the results into output
    if os.path.isdir(output):
        try:
            shutil.rmtree(output)
        except EnvironmentError:
            sys.exit("Unable to remove directory: "+output)
    
    # create new directory
    try:
        os.mkdir(output)
    except EnvironmentError:
        sys.exit("Unable to create directory: "+output)
def write_output(clusters, output, df_distance , m2clust_scores = None, sorted_keys = None):
    
    f = open(output+"/m2clust.txt", 'w')
    print ("There are %s clusters" % len(clusters))
    
    
    metadata_order = ''
    if sorted_keys is not None:
        zipped = list(zip(clusters, m2clust_scores[sorted_keys[0]]))
        zipped.sort(key = lambda t: t[1],  reverse=True)
        importance_order = numpy.argsort(m2clust_scores[sorted_keys[0]])[::-1]
        for key in sorted_keys:
            metadata_order +='\t'+ key
    else:
        importance_order =  range(len(clusters))
    f.write("cluster" + "\t" + "members" + metadata_order + '\n')
    for i in importance_order:
        f.write('C'+str(i+1)+"\t")
        features = clusters[i].pre_order(lambda x: x.id)
        feature_names = [df_distance.index[val] for val in features]
        for j in range(len(feature_names)-1):
            f.write("%s;" % feature_names[j])           
        f.write("%s" % feature_names[len(feature_names)-1])
        if m2clust_scores is not None:
            for key in sorted_keys:
                f.write("\t" + str(m2clust_scores[key][i]))
        f.write("\n")
    #print c_medoids
    '''if not df_data.empty:
        df_data.loc[c_medoids, :].to_csv(path_or_buf=output+'/medoids.txt', sep='\t' )'''
        
    print ("Output is written in %s" % output)
    
def cluster2dict(clusters, df_distance):
    clusters_dic = {}
    for i in range(len(clusters)):
        features = clusters[i].pre_order(lambda x: x.id)
        #feature_names = [df_distance.index[val] for val in features]
        clusters_dic['C'+str(i+1)] = features
    return clusters_dic
    