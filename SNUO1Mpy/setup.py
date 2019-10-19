"""
SNUO STX 4k CCD (photometry)
"""

from setuptools import setup, find_packages

setup_requires = []
install_requires = ['numpy>=1.14',
                    'scipy>=0.17',
                    'astropy >= 2.0']

classifiers = ["Intended Audience :: Science/Research",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3.6"]

setup(
    name="snuo1mpy",
    version="0.0.1.dev",
    author="Yoonsoo P. Bach",
    author_email="dbstn95@gmail.com",
    description="Data reduction package for the 1-m telescope at SNUO",
    license="MIT",
    keywords="",
    url="https://github.com/ysBach/SNU1Mpy",
    classifiers=classifiers,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires )
