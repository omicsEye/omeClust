import sys
import unittest
import itertools

from omeClust import omeClust

try:
    from numpy import array
except ImportError:
    sys.exit("Please install numpy")


class TestomeClustDistanceFunctions(unittest.TestCase):
    """
    Test the functions found in omeClust
    """
        
    def test_import(self):
        """
        Test the normalized mututal information function
        """
        pass

