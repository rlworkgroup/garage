"""setuptools based setup module."""
from setuptools import find_packages
from setuptools import setup

TF_VERSION = '<1.16,>=1.15.0'
GYM_VERSION = '==0.12.4'

# Required dependencies
REQUIRED = [
    # Please keep alphabetized
    'akro==0.0.6',
    'cached_property',
    'click',
    'cloudpickle',
    'cma==2.7.0',
    'dowel==0.0.2',
    'gym[atari,box2d,classic_control]' + GYM_VERSION,
    'joblib<0.13,>=0.12',
    'matplotlib',
    'numpy>=1.14.5',
    'psutil',
    # Pyglet 1.4.0 introduces some api change which breaks some
    # gym environments
    # See: https://github.com/openai/gym/issues/1588
    'pyglet<1.4.0,>=1.3.0',
    'pyprind',
    'python-dateutil',
    'torch==1.3.0',
    'ray',
    'scikit-image',
    'scipy',
    'tensorflow' + TF_VERSION,
    'tensorflow-probability',
    'torchvision==0.4.1'
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
    'dm_control @ https://api.github.com/repos/deepmind/dm_control/tarball/7a36377879c57777e5d5b4da5aae2cd2a29b607a',  # noqa: E501
]

EXTRAS['all'] = list(set(sum(EXTRAS.values(), [])))

# dependencies for using gpu, not included in 'all'
EXTRAS['gpu'] = ['tensorflow-gpu' + TF_VERSION]

# Development dependencies (*not* included in 'all')
EXTRAS['dev'] = [
    # Please keep alphabetized
    'baselines @ https://api.github.com/repos/openai/baselines/tarball/f2729693253c0ef4d4086231d36e0a4307ec1cb3',  # noqa: E501
    'flake8',
    'flake8-docstrings>=1.5.0',
    'flake8-import-order',
    'gtimer',
    'pandas',
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
    'rlkit @ git+https://github.com/vitchyr/rlkit/@1d469a509b797ca04a39b8734c1816ca7d108fc8',  # noqa: E501
    'seaborn',
    'sphinx',
    'sphinx_rtd_theme',
    'yapf==0.28.0',
]

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
