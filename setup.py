"""setuptools based setup module."""
from setuptools import find_packages
from setuptools import setup

TF_VERSION = '<1.15,>=1.14.0'
GYM_VERSION = '==0.12.4'

# Required dependencies
required = [
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
    'torch==1.1.0',
    'ray',
    'scikit-image',
    'scipy',
    'tensorflow' + TF_VERSION,
    'tensorflow-probability<0.8.0,>=0.7.0',  # for tensorflow 1.12
    'torchvision==0.3.0'
]

# Dependencies for optional features
extras = {}

extras['mujoco'] = [
    'mujoco-py<2.1,>=2.0',
    'gym[all]' + GYM_VERSION,
]

extras['dm_control'] = [
    # dm_control throws an error during install about not being able to
    # find a build dependency (absl-py). Later pip executes the `install`
    # command again and the install succeeds because absl-py has been
    # installed. This is stupid, but harmless.
    'dm_control @ https://api.github.com/repos/deepmind/dm_control/tarball/7a36377879c57777e5d5b4da5aae2cd2a29b607a',  # noqa: E501
]

extras['all'] = list(set(sum(extras.values(), [])))

# dependencies for using gpu, not included in all
extras['gpu'] = ['tensorflow-gpu' + TF_VERSION]

# Development dependencies (*not* included in "all")
extras['dev'] = [
    # Please keep alphabetized
    'baselines @ https://api.github.com/repos/openai/baselines/tarball/f2729693253c0ef4d4086231d36e0a4307ec1cb3',  # noqa: E501
    'flake8',
    'flake8-docstrings==1.4.0',
    'flake8-import-order',
    'gtimer',
    'pandas',
    'pep8-naming==0.7.0',
    'pre-commit',
    'pydocstyle<4.1,>=4.0',
    'pylint==1.9.2',
    'pytest>=3.6',  # Required for pytest-cov on Python 3.6
    'pytest-cov',
    'pytest-xdist',
    'recommonmark',
    'rlkit @ git+https://github.com/vitchyr/rlkit/@1d469a509b797ca04a39b8734c1816ca7d108fc8',  # noqa: E501
    'seaborn',
    'sphinx',
    'sphinx_rtd_theme',
    'yapf==0.28.0',
]

with open('README.md') as f:
    readme = f.read()

# Get the package version dynamically
with open('VERSION') as v:
    version = v.read().strip()

setup(
    name='garage',
    version=version,
    author='Reinforcement Learning Working Group',
    description='A framework for reproducible reinforcement learning research',
    url='https://github.com/rlworkgroup/garage',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    scripts=['scripts/garage'],
    python_requires='>=3.5',
    install_requires=required,
    extras_require=extras,
    license='MIT',
    long_description=readme,
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
