# m2clust: multi-resolution clustering of omics data #

**m2clust** is a clustering method that detects
clusters of features using omics data and scores metadata 
(resolution score) based on their influences in clustering.
The similarity of features within each cluster can be 
different (different resolution). Resolution of similarity score takes to 
account not only similarity between measurements and 
also the structure in a hierarchical structure of data and 
number of features which group together.

---

**Citation:**

Rahnavard A. et al, **m2clust: multi-resolution clustering of omics data** . https://github.com/omicsEye/m2clust/

----

* Please see the [Workshop](https://github.com/omicsEye/m2clust/wiki/Workshop) for a one hour workshop.

----
# mclust user manual

## Contents ##
* [Features](#features)
* [m2clust](#m2clust)
    * [mclust approach](#mclust-approach)
    * [Requirements](#requirements)
    * [Installation](#installation)
* [Getting Started with mclust](#getting-started-with-mclust)
    * [Test mclust](#test-mclust)
    * [Options](#options) 
    * [Input](#input)
    * [Output](#output)  
* [How to run](#how-to-run)
    * [Basic usage](#basic-usage)
    * [Setting for cluster resolution](#setting-for-cluster-resolution)
    * [Demo runs](#demo-runs)
* [Guides to mclustviz for visuzlaization](#guides-to-m2clustviz-for-visualiazation)
* [Synthetic clusters](#synthetic-clusters)
* [Output files](#output-files)
    1. [Cluster file](#clsters-file)
    2. [Distance table](#distance-table)
* [Result plots](#result-plots)
    1. [PCoA plot](#pcoa-plot)
    2. [MDS plot](#MDS-plot)
    3. [t-SNE plot](#t-sne-plot)
* [Configuration](#markdown-header-configuration)
* [Tutorials for distance calculation](#tutorials-for-distance-calculation)
    * [Distance between sequencing alignments](#distance-between-sequencing-alignments)
    * [Distance using correlation](#Distance-using-correlation)
    * [Distance using entropy](#distance-using-entropy)
* [Tools](#markdown-header-tools)
    * [m2clust synthetic paired datasets generator](#m2clust-synthetic-paired-datasets-generator)
    * [m2clust Python API](#m2clust-python-api)
* [FAQs](#markdown-header-faqs)
* [Complete option list](#markdown-header-complete-option-list)
------------------------------------------------------------------------------------------------------------------------------
# Features #
1. Generality: m2clust uses distance matrix as input

2. A simple user interface (single command driven flow)
    * The user only needs to provide a paired dataset
    
# m2clust #
## m2clust appraoch ##
![overview.png](https://github.com/omicsEye/m2clust/docs/overviwe.png)
## REQUIREMENTS ##
* [Matplotlib](http://matplotlib.org/)
* [Python 3.*](https://www.python.org/download/releases/)
* [Numpy 1.9.*](http://www.numpy.org/)
* [Pandas (version >= 0.18.1)](http://pandas.pydata.org/getpandas.html)

## INSTALLATION ##
```
$ sudo pip3 install m2clust
```
------------------------------------------------------------------------------------------------------------------------------

# Getting Started with mclust #
## TEST m2clust ##

To test if m2clust is installed correctly, you may run the following command in the terminal:

```
#!cmd

m2clust -h

```

Which yields m2clust command line options


## Options ##

```
#!python
usage: m2clust [-h] [-i INPUT] -o OUTPUT [-m SIMILARITY] [--metadata METADATA]
               [-n ESTIMATED_NUMBER_OF_CLUSTERS] [--size-to-plot SIZE_TO_PLOT]
               [-c LINKAGE_METHOD] [--plot] [--resolution {high,medium,low}]
               [-v]

Multi-resolution clustering using hierarchical clustering and Silhouette score.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        the input file D*N, Rows: D features and columns: N samples OR 
                        a distance matrix file D*D (rows and columns should be the same and in the same order) 
                         
  -o OUTPUT, --output OUTPUT
                        the output directory
  -m SIMILARITY, --similarity SIMILARITY
                        similarity measurement {default spearman, options: spearman, nmi, ami, dmic, mic, pearson, dcor}
  --metadata METADATA   Rows are features and each column is a metadata
  -n ESTIMATED_NUMBER_OF_CLUSTERS, --estimated_number_of_clusters ESTIMATED_NUMBER_OF_CLUSTERS
                        estimated number of clusters
  --size-to-plot SIZE_TO_PLOT
                        Minimum size of cluster to be plotted
  -c LINKAGE_METHOD, --linkage_method LINKAGE_METHOD
                        linkage clustering method method {default = single, options average, complete
  --plot                dendrogram plus heatmap
  --resolution {high,medium,low}
                        Resolution c .         Low resolution is good when clusters are well separated clusters.
  -v, --verbose         additional output is printed
```


## Input ##

* ``-i or --input:`` a distance matrix.
* ``--output-folder``: a folder containing all the output files
* ``--resolution``: a resolution to be used for clustering {low or high}

Th input is a tab-delimited text file of `n * n` where `n` is number of features 
(e.g. metabolites, stains, microbial species, individuals).
  


## Output ##

the main output is the `m2clust.txt` a a tab-delimited text file that each row is a cluster with following columns.
* cluster: is a cluster name starts with C	
* members: members of a cluster	
* resolution_score: an score defined by 	
* Meta1: if metadata is provides this is the first metadata that is enriched in cluster and
is reported as most influential metadata on clusters structure. 	
* Meta2: the second most 
influential metadata. (Metadata2 is a name of a column in metadata if if it is provided).

Below is an example output `m2clust.txt` file:
```
cluster  |  members                                  |  resolution_score  |  Meta1        |  Meta2
---------|-------------------------------------------|--------------------|---------------|-------------
C7       |  A44;A48;A42;A46;A49;A41;A43;A47;A45;A50  |  0.337590449       |  0.311316112  |  0.29595993
C1       |  A26;A24;A21;A23;A20;A28;A25;A22;A27      |  0.299340924       |  0.276096463  |  0.276096463
C4       |  A35;A37;A33;A38;A40;A36;A34;A39          |  0.271570917       |  0.254305428  |  0.244961184
C6       |  A2;A7;A5;A4;A6;A1;A3                     |  0.246264653       |  0.231982388  |  0.229761528
C2       |  A8;A9;A10;A13;A12;A11;A14                |  0.239955783       |  0.217158357  |  0.226375741
C3       |  A16;A19;A15;A17;A18                      |  0.181968762       |  0.17405081   |  0.172177829
C5       |  A32;A31;A29;A30                          |  0.149117609       |  0.143758382  |  0.142056564
```
*   File name: `` $OUTPUT_DIR/m2clust.txt ``
*   This file details the clusters. Features are grouped in clusters.
*    **```cluster```**: a column contains clusters names that each cluster name starts with `C` following with a number.
*    **```members```**: has one or more features that participate in the cluster.
*    **```resolution_score```**: this value is corresponding to `binary silhouette score` introduced in this work.
*    **```Meta1```**: 
*    **```Meta2```**:
## Output files ##
1. [###](#PCoA)
2. [###](###)
3. [###](####)
 4. [###](####)

### 1. First dataset heatmap ###
![](http:// =15x)

### 2. mclust ordination plots ###
![](http://.png =15x)

*   File name: `` $OUTPUT_DIR/###.pdf ``
*   This file has a 
*   ###

# Guides to mclustviz for visuzlaization #


* **Basic usage:** `$  mclustviz $DISTANCE_MATRIX.txt /path-to-mclust-output/mclust.txt --metadata $META_DATA.txt --shapeby $METADATA1 -o /path-to-mclust-output/`
* `$DISTANCE_MATRIX.txt` = an distance matrix that used for clustering 
* `mclust.txt` = an mclust output which assigns features to clusters
* `$META_DATA.txt`: is metadata file which contains metadat for features
* `$META_DATA1`: is a metadata in the metadata file to be used for shaping poins in the ordination plot
* Run with `-h` to see additional command line options

Produces a set of ordination plots for features colored by computational clusters and shaped by metadata.

```
usage: mclustviz [-h] [--metadata METADATA] [--shapeby SHAPEBY] -o OUTPUT
                 [--size-to-plot SIZE_TO_PLOT]
                 adist clusters

mclust visualization script.

positional arguments:
  adist                 the input file D*N, Rows: D features and columns: N samples OR 
                        a distance matrix file D*D (rows and columns should be the same and in the same order) 
                         
  clusters              the input file D*N, Rows: D features and columns: N samples OR 
                        a distance matrix file D*D (rows and columns should be the same and in the same order) 
                         

optional arguments:
  -h, --help            show this help message and exit
  --metadata METADATA   metadata
  --shapeby SHAPEBY     the input file D*N, Rows: D features and columns: N samples OR 
                        a distance matrix file D*D (rows and columns should be the same and in the same order) 
                         
  -o OUTPUT, --output OUTPUT
                        the output directory
  --size-to-plot SIZE_TO_PLOT
                        Minimum size of cluster to be plotted
```

![t-SNE a plot of strains for microbial species in the expanded Human Microbiome Project (HMP1-II)](https://github.com/omicsEye/mclust/blob/master/img/t-SNE_plot.png)



### Quick start ###

* Installation

*m2clust* is implemented in python and packaged and available
via PyPi. Run the following command to get it installed (use `sudo`
to install it for all users or use --user and provide a path with write access) 

``
$ sudo pip3 install m2clust
`` 
* Input data 

The input data is a distance matrix of feature `n*n` 
where `n` is the number of features.
optional input is a metadata table `n*m` where 
`n` is the number of features and `m` is the number of metadata

* How to run?


``
$ m2clust -i synthetic_demo/adist.txt -o demo_output
``

if metadata is available then use the following command:

``
$ m2clust -i synthetic_demo/adist.txt -o demo_output --metadata synthetic_demo/metadata.txt  --plot
``

`--plot` is optional to generate a heatmap with 
deprogram of the data 

`--metadata` is optional to shape the clusters with 
highest influence in clusters.

* How to run from script?

``
$ python3
`` 

``
from m2clust import m2clust
``

``
m2clust.m2clust(data='/path-to/adist.txt', metadata='/path-to/metadata.txt', 
                output_dir='m2clust_output')
``
* output
1. `m2clust.txt` contains cluster, their members,
and metadata resolution score sorted 
from highest to lowest score.

* [Learn more about details of options](https://github.com/omicsEye/m2clust/wiki)

### Demo run using synthetic data ###
1. Download the input:
[Distance matrix](m2clust_demo/synthetic_data/adist.txt) and
[metadata](m2clust_demo/synthetic_data/metadata.txt))
2. Run m2clust in command line with input
``$ m2clust -i synthetic_demo/adist.txt -o demo_output --metadata synthetic_demo/metadata.txt --plot``
3. Check your output folder

Here we show the PCoA and DMS plot as one the representative 
visualization of the results. 

<img src="m2clust_demo/output/PCoA_plot.png" height="35%" width="35%">
<img src="m2clust_demo/output/MDS_plot.png" height="35%" width="35%">

### Real world example ###

Please see the wiki for real-world example including: 
gene expression, microbial species stains, and metabolite profiles.

<img src="m2clust_demo/output/PCoA_plot.png" height="35%" width="35%">
<img src="m2clust_demo/output/MDS_plot.png" height="35%" width="35%">
<img src="m2clust_demo/output/PCoA_plot.png" height="35%" width="35%">

Please see the [Workshop](https://github.com/omicsEye/m2clust/wiki) for the data, their description.

# Synthetic clusters #

```buildoutcfg
$ python3
from  m2clust import cluster_generator
from  m2clust import dataprocess
nX = 100
nY = 100 
nSamples = 50
 X,Y,A = cluster_generator.circular_block(nSamples = nSamples, nX =nX, nY = nY, nBlocks =5, noiseVar = 0.1,
... blockIntraCov = 0.3, offByOneIntraCov = 0.0,
... blockInterCov = 0.2, offByOneInterCov = 0.0,
... holeCov = 0.3, holeProb = .25)

# wite file
dataprocess.write_table(X, name= '/your-file-path/' + 'X_'+ str(nSamples) + '_' + str(nX)+ '.txt', prefix="Feature")

dataprocess.write_table(Y, name= '/your-file-path/' + 'Y_'+ str(nSamples) + '_' + str(nY)+ '.txt', prefix="Feature")
rowheader = ['Feature'+ str(i) for i in range(0, nX)]
colheader = ['Feature'+ str(i) for i in range(0, nY)]

dataprocess.write_table(A, name= '/your-file-path/' + 'A_'+ str(nX) + '_' + str(nY) +'.txt', prefix="Feature", colheader = colheader, rowheader = rowheader)
```
`circular_block` function returns two datasets `X` and `Y`, and also 
`A` matrix for relationships between features among these two datasets.

Here is a description for parameters of the function for properties of 
the datasets and spiked relationship within and between datasets:
* `nSample`: number of samples in each datasets (appers as columns)
* `nX`: number of features in each datasets (appears as rows of X)
* `nY`: number of features in each datasets (appears as rows of Y)
* `nBlocks`: number of clusters in each dataset
* `noiseVar`: noise variable between [0.0..1.0], 0.0 refers to no noise
* `blockIntraCov`: specifies covariance between features within a cluster
* `offByOneIntraCov`: 
* `blockInterCov`: specifies covariance between features among clusters between datasets
* `offByOneInterCov`:
* `holeCov`:
* `holeCov`: 
* `holeProb`: 


### Support ###

* Please submit your questions or issues with the software at [Issues tracker](https://github.com/omicsEye/m2clust/issues).


