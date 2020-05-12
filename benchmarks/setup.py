"""Setup script for garage benchmarking scripts.

This package is generally not needed by users of garage.
"""

from setuptools import find_packages, setup

setup(name='garage_benchmarks',
      packages=find_packages(where='src'),
      package_dir={'': 'src'})
