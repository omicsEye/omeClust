import sys
import unittest

from halla import stats

try:
    from numpy import array
except ImportError:
    sys.exit("Please install numpy")


class Testm2clustStatsFunctions(unittest.TestCase):
    """
    Test the functions found in m2clust
    """

    def test_import(self):
        """
        Test the normalized mututal information function
        """
        pass
