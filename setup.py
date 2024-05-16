import sys
try:
    from setuptools import setup, find_packages
except ImportError:
    sys.exit("Please install setuptools.")

import os

try:
    from urllib.request import urlretrieve

    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ]
except ImportError:
    from urllib import urlretrieve
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ]

VERSION = "1.1.10"
AUTHOR = "Ali Rahnavard"
AUTHOR_EMAIL = "gholamali.rahnavard@gmail.com"
MAINTAINER = "Ali Rahnavard"
MAINTAINER_EMAIL = "gholamali.rahnavard@gmail.com"

# try to download the bitbucket counter file to count downloads
# this has been added since PyPI has turned off the download stats
# this will be removed when PyPI Warehouse is production as it
# will have download stats
COUNTER_URL = "https://github.com/omicsEye/omeClust/blob/master/README.md"
counter_file = "README.md"
if not os.path.isfile(counter_file):
    print("Downloading counter file to track omeClust downloads" +
          " since the global PyPI download stats are currently turned off.")
    try:
        pass  # file, headers = urlretrieve(COUNTER_URL,counter_file)
    except EnvironmentError:
        print("Unable to download counter")
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="omeClust",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    version=VERSION,
    license="MIT",
    description="omeClust: multi-resolution clustering",
    long_description="omeClust provides an elegant clustering approach " + \
                     "to find clusters in data sets with different density and resolution.",
    url="http://github.com/omicsEye/omeClust",
    keywords=['clustering', 'multi-resolution', 'hierarchically'],
    platforms=['Linux', 'MacOS', "Windows"],
    classifiers=classifiers,
    # long_description=open('readme.md').read(),
    data_files=[('.', ['requirements.txt'])],
    install_requires=required,
    # cycler==0.11.0
    # joblib==1.1.0
    # kiwisolver==1.3.2
    # matplotlib==3.4.3
    # networkx==2.6.3
    # numpy==1.21.4
    # pandas==1.3.4
    # Pillow==8.4.0
    # pyparsing==3.0.6
    # python-dateutil==2.8.2
    # pytz==2021.3
    # scikit-learn==1.0.1
    # scipy==1.7.2
    # six==1.16.0
    # threadpoolctl==3.0.0
    # Community == 1.0.0b1
    # latex >= 0.0.1
    # Cython >= 0.29.2
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'omeClust = omeClust.omeClust:main',
            'omeClustviz = omeClust.viz:main',
            'omeClust_test = omeClust.tests.omeClust_test:main'
        ]},
    test_suite='omeClust.tests.omeClust_test',
    zip_safe=False
)
