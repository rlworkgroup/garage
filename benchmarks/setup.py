"""Setup script for garage benchmarking scripts.

This package is generally not needed by users of garage.
"""
import os

from setuptools import find_packages, setup

GARAGE_GH_TOKEN = os.environ.get('GARAGE_GH_TOKEN') or 'git'

REQUIRED = [
    # Please keep alphabetized
    'baselines @ https://{}@api.github.com/repos/openai/baselines/tarball/ea25b9e8b234e6ee1bca43083f8f3cf974143998'.format(GARAGE_GH_TOKEN),  # noqa: E501
    'matplotlib',
]  # yapf: disable

setup(name='garage_benchmarks',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      install_requires=REQUIRED)
