import sys
import unittest

from halla import stats

try:
    from numpy import array
except ImportError:
    sys.exit("Please install numpy")


class TestomeClustStatsFunctions(unittest.TestCase):
    """
    Test the functions found in omeClust
    """

    def test_import(self):
        """
        Test the normalized mututal information function
        """
        pass
