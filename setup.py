"""setuptools based setup module."""
import os

from setuptools import find_packages
from setuptools import setup

GARAGE_GH_TOKEN = os.environ.get('GARAGE_GH_TOKEN') or 'git'
GYM_VERSION = '==0.15.4'

# Required dependencies
REQUIRED = [
    # Please keep alphabetized
    'akro',
    'cached_property',
    'click>=2.0',
    'cloudpickle',
    'cma==2.7.0',
    'dowel==0.0.3',
    'gym[atari,box2d,classic_control]' + GYM_VERSION,
    'joblib<0.13,>=0.12',
    'numpy>=1.14.5',
    'psutil',
    # Pyglet 1.4.0 introduces some api change which breaks some
    # gym environments
    # See: https://github.com/openai/gym/issues/1588
    'pyglet<1.4.0,>=1.3.0',
    'python-dateutil',
    'ray',
    'scikit-image',
    'scipy',
    'setproctitle>=1.0',
    'tensorflow',
    'tensorflow-probability',
    'torch>=1.0.0,<1.5.0',
    'torchvision>=0.2.1,<0.6.0',
]

# Dependencies for optional features
EXTRAS = {}

EXTRAS['mujoco'] = [
    'mujoco-py<2.1,>=2.0',
    'gym[all]' + GYM_VERSION,
]

EXTRAS['dm_control'] = [
    # dm_control throws an error during install about not being able to
    # find a build dependency (absl-py). Later pip executes the `install`
    # command again and the install succeeds because absl-py has been
    # installed. This is stupid, but harmless.
    'dm_control==0.0.300771433',
]

EXTRAS['all'] = list(set(sum(EXTRAS.values(), [])))

# Development dependencies (*not* included in 'all')
EXTRAS['dev'] = [
    # Please keep alphabetized
    'flake8',
    'flake8-docstrings>=1.5.0',
    'flake8-import-order',
    'metaworld @ https://{}@api.github.com/repos/rlworkgroup/metaworld/tarball/861ae8d8c4bef80a7ed86f47f47acaa494d4ab77'.format(GARAGE_GH_TOKEN),  # noqa: E501
    'pep8-naming==0.7.0',
    'pre-commit',
    'pycodestyle>=2.5.0',
    'pydocstyle>=4.0.0',
    'pylint>=2.4.3',
    'pytest>=4.5.0',  # Required for strict-markers
    'pytest-cov',
    'pytest-timeout',
    'pytest-xdist',
    'recommonmark',
    'sphinx',
    'sphinx_rtd_theme',
    'yapf==0.28.0',
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
    python_requires='>=3.5',
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
)
