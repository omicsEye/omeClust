# m2clust: multi-resolution clustering of omics data #

**m2clust** is a clustering method that detects
clusters of features using omics data and scores metadata 
(resolution score) based on their influences in clustering.
The similarity of features within each cluster can be 
different (different resolution). Resolution of similarity score takes to 
account not only similarity between measurements and 
also the structure in a hierarchical structure of data and 
number of features which group together.

**Citation:** Rahnavard A. et al, *m2clust: multi-resolution clustering of omics data*

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

Please see the wiki for real world example including: 
gene expression, microbial species stains, and metabolite profiles.

<img src="m2clust_demo/output/PCoA_plot.png" height="35%" width="35%">
<img src="m2clust_demo/output/MDS_plot.png" height="35%" width="35%">
<img src="m2clust_demo/output/PCoA_plot.png" height="35%" width="35%">

Please see the [Wiki](https://github.com/omicsEye/m2clust/wiki) for the data, their description.


### Support ###

* Please submit your questions or issues with the software at [Issues tracker](https://github.com/omicsEye/m2clust/issues).
