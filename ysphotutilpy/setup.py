"""
"""

from setuptools import setup, find_packages

setup_requires = []
install_requires = ['numpy',
                    'pandas',
                    'astropy >= 2.0',
                    'ccdproc >= 1.3',
                    'matplotlib >= 2',
                    'astroquery >= 0.3.9']

classifiers = ["Intended Audience :: Science/Research",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3.6"]

setup(
    name="ysphotutilpy",
    version="0.0.2.dev",
    author="Yoonsoo P. Bach",
    author_email="dbstn95@gmail.com",
    description="",
    license="",
    keywords="",
    url="",
    classifiers=classifiers,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires)
