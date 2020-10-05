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
            sys.exit("Unable to remove directory: " + output)

    # create new directory
    try:
        os.mkdir(output)
    except EnvironmentError:
        sys.exit("Unable to create directory: " + output)


def write_output(clusters, output, df_distance, enrichment_scores=None, sorted_keys=None):
    f = open(output + "/clusters.txt", 'w')
    print("There are %s clusters" % len(clusters))

    metadata_order = ''
    if sorted_keys is not None:
        zipped = list(zip(clusters, enrichment_scores[sorted_keys[0]]))
        zipped.sort(key=lambda t: t[1], reverse=True)
        importance_order = numpy.argsort(enrichment_scores[sorted_keys[0]])[::-1]
        for key in sorted_keys:
            metadata_order += '\t' + key
    else:
        importance_order = range(len(clusters))
    f.write("Cluster" + "\t" + "Members" + metadata_order + '\n')
    for i in importance_order:
        f.write('C' + str(i + 1) + "\t")
        features = clusters[i].pre_order(lambda x: x.id)
        feature_names = [df_distance.index[val] for val in features]
        for j in range(len(feature_names) - 1):
            f.write("%s;" % feature_names[j])
        f.write("%s" % feature_names[len(feature_names) - 1])
        if enrichment_scores is not None:
            for key in sorted_keys:
                f.write("\t" + str(enrichment_scores[key][i]))
        f.write("\n")
    # print c_medoids
    '''if not df_data.empty:
        df_data.loc[c_medoids, :].to_csv(path_or_buf=output+'/medoids.txt', sep='\t' )'''

    print("Output is written in %s" % output)


def cluster2dict(clusters):
    clusters_dic = {}
    for i in range(len(clusters)):
        features = clusters[i].pre_order(lambda x: x.id)
        clusters_dic['C' + str(i + 1)] = features
    return clusters_dic

def feature2cluster(clusters, D):
    feature2cluster_dic = {}
    for i in range(len(clusters)):
        features = clusters[i].pre_order(lambda x: x.id)
        feature_names = [D.index[val] for val in features]
        for feature in feature_names:
            feature2cluster_dic[feature] = 'C' + str(i + 1) #'C' + str(i + 1)
    return feature2cluster_dic


def write_table(data=None, name=None, rowheader=None, colheader=None, prefix="label", col_prefix=None, corner=None,
                delimiter='\t'):
    """
    wite a matrix of data in tab-delimated format file

    input:
    data: a 2 dimensioal array of data
    name: includes path and the name of file to save
    rowheader
    columnheader

    output:
    a file tabdelimated file
    """

    if data is None:
        print("Null input for writing table")
        return
    f = open(name, 'w')
    # row numbers as header
    if colheader is None:
        if corner is None:
            f.write(delimiter)
        else:
            f.write(corner)
            f.write(delimiter)
        if col_prefix is None:
            col_prefix = 'S'
        for i in range(len(data[0])):
            f.write(col_prefix + str(i))
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    elif len(colheader) == len(data[0]):
        if corner is None:
            f.write(delimiter)
        else:
            f.write(corner)
            f.write(delimiter)
        for i in range(len(data[0])):
            f.write(colheader[i])
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    else:
        print("The label list in not matched with the data size")
        sys.exit()

    for i in range(len(data)):
        if rowheader is None:
            f.write(prefix + str(i))
            f.write(delimiter)
        else:
            f.write(rowheader[i])
            f.write(delimiter)
        for j in range(len(data[i])):
            f.write(str(data[i][j]))
            if j < len(data[i]) - 1:
                f.write(delimiter)
        f.write('\n')
    f.close()
