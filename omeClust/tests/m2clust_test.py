""" omeClust_test.py : Test discovery module for omeClust. """

import os
import unittest


def main():
    directory_of_tests=os.path.dirname(os.path.realpath(__file__))
    basic_suite = unittest.TestLoader().discover(directory_of_tests,pattern='basic_tests_*.py')
    advanced_suite = unittest.TestLoader().discover(directory_of_tests, pattern='advanced_tests_*.py')
    full_suite = unittest.TestSuite([basic_suite, advanced_suite])
    return full_suite
