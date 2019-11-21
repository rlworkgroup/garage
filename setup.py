from setuptools import find_packages
from setuptools import setup

# Get the package version dynamically
with open('VERSION') as v:
    version = v.read().strip()

setup(
    name='rlgarage',
    version=version,
    author='Reinforcement Learning Working Group',
    description='A framework for reproducible reinforcement learning research',
    url='https://github.com/rlworkgroup/garage',
    packages=[
        package for package in find_packages() if package.startswith('garage')
    ],
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
)
