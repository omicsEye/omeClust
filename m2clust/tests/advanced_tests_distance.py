import sys
import unittest
import itertools

from m2clust import m2clust

try:
    from numpy import array
except ImportError:
    sys.exit("Please install numpy")


class Testm2clustDistanceFunctions(unittest.TestCase):
    """
    Test the functions found in m2clust
    """
        
    def test_import(self):
        """
        Test the normalized mututal information function
        """
        pass

