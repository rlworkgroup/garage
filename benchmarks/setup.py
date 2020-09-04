"""Setup script for garage benchmarking scripts.

This package is generally not needed by users of garage.
"""
import os

from setuptools import find_packages, setup

GARAGE_GH_TOKEN = os.environ.get('GARAGE_GH_TOKEN') or 'git'

REQUIRED = [
    # Please keep alphabetized
    'google-cloud-storage',
    'matplotlib'
]  # yapf: disable

setup(name='garage_benchmarks',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      install_requires=REQUIRED,
      include_package_data=True,
      entry_points='''
              [console_scripts]
              garage_benchmark=garage_benchmarks.run_benchmarks:cli
          ''')
