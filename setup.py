"""setuptools based setup module."""
import os

from setuptools import find_packages, setup

GARAGE_GH_TOKEN = os.environ.get('GARAGE_GH_TOKEN') or 'git'
GYM_VERSION = '0.17.2'

# Required dependencies
REQUIRED = [
    # Please keep alphabetized
    'akro',
    'click>=2.0',
    'cloudpickle==1.3',
    'cma==2.7.0',
    'dowel==0.0.3',
    f'gym[atari,box2d,classic_control]=={GYM_VERSION}',
    'numpy>=1.14.5',
    'psutil',
    'python-dateutil',
    'ray',
    'scikit-image',
    'scipy',
    'setproctitle>=1.0',
    'tensorflow>=1.14',
    'tensorflow-probability>=0.11.0',
    'torch>=1.0.0,!=1.5.0,<1.6.0',
    'torchvision>=0.2.1,<0.7.0',
]

# Dependencies for optional features
EXTRAS = {}

EXTRAS['mujoco'] = [
    'mujoco-py<2.1,>=2.0',
    f'gym[all]=={GYM_VERSION}',
]

EXTRAS['dm_control'] = [
    # dm_control throws an error during install about not being able to
    # find a build dependency (absl-py). Later pip executes the `install`
    # command again and the install succeeds because absl-py has been
    # installed. This is stupid, but harmless.
    'dm_control',
]

EXTRAS['bullet'] = ['mpi4py', 'pybullet>=2.8.7']

EXTRAS['all'] = list(set(sum(EXTRAS.values(), [])))

# Development dependencies (*not* included in 'all')
EXTRAS['dev'] = [
    # Please keep alphabetized
    'flake8',
    'flake8-docstrings>=1.5.0',
    'flake8-import-order',
    f'metaworld @ https://{GARAGE_GH_TOKEN}@api.github.com/repos/rlworkgroup/metaworld/tarball/861ae8d8c4bef80a7ed86f47f47acaa494d4ab77',  # noqa: E501
    'isort>=4.3.21,<5.0.0',
    'pep8-naming==0.7.0',
    'pre-commit',
    'pycodestyle>=2.5.0',
    'pydocstyle>=4.0.0',
    'pylint>=2.5.3',
    'pytest>=4.5.0',  # Required for strict-markers
    'pytest-cov',
    'pytest-timeout',
    'pytest-xdist',
    'recommonmark',
    'sphinx',
    'sphinx-autoapi>=1.4.0',
    'sphinx_rtd_theme',
    'sphinxcontrib-bibtex',
    'yapf==0.30.0',
]  # yapf: disable

with open('README.md') as f:
    README = f.read()

# Get the package version dynamically
with open('VERSION') as v:
    VERSION = v.read().strip()

setup(
    name='garage',
    version=VERSION,
    author='Reinforcement Learning Working Group',
    description='A toolkit for reproducible reinforcement learning research',
    url='https://github.com/rlworkgroup/garage',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    scripts=['scripts/garage'],
    python_requires='>=3.6',
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    license='MIT',
    long_description=README,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
)
