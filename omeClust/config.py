version = '1.1.3'
__description__ = """
 omeClust for multi-resolution clustering
"""

__doc__ = __doc__
__version__ = version
__author__ = ["Ali Rahnavard"]
__contact__ = "gholamali.rahnavard@gmail.com"

keys_attribute = ["__description__", "__version__", "__author__", "__contact__", "clustering", "multi-resolution"]

# default Parameters
similarity_method = 'spearman'
diatance_metric = 'spearman'  # euclidean'
data = None
metadata = None
resolution = 'low'
output_dir = 'omeClust_output'
estimated_number_of_clusters = 2
linkage_method = 'average'
plot = False
size_to_plot = 3
enrichment_method = 'nmi'

# output directort
output_dir = './'
