from setuptools import find_packages
from setuptools import setup

# Required dependencies
required = [
    # Please keep alphabetized
    'akro',
    'boto3',
    'cached_property',
    'cloudpickle',
    'cma==1.1.06',
    # dm_control throws an error during install about not being able to
    # find a build dependency (absl-py). Later pip executes the `install`
    # command again and the install succeeds because absl-py has been
    # installed. This is stupid, but harmless.
    'dm_control @ https://api.github.com/repos/deepmind/dm_control/tarball/92f9913013face0468442cd0964d5973ea2089ea',  # noqa: E501
    'gym[all]==0.10.11',
    'hyperopt',
    'jsonmerge',
    'joblib<0.13,>=0.12',
    'matplotlib',
    'mujoco-py<1.50.2,>=1.50.1',
    'numpy==1.14.5',
    'pandas',
    'polling',
    'protobuf',
    'psutil',
    'pyprind',
    'python-dateutil',
    'scikit-image',
    'scipy',
    'tabulate',
    'tensorboardX',
    'tensorflow<1.13,>=1.12.0',
    'tensorflow-probability<0.6.0,>=0.5.0',  # for tensorflow 1.12
]

# Dependencies for optional features
extras = {}
extras['all'] = list(set(sum(extras.values(), [])))

# Development dependencies (*not* included in "all")
extras['dev'] = [
    # Please keep alphabetized
    'baselines @ https://api.github.com/repos/openai/baselines/tarball/f2729693253c0ef4d4086231d36e0a4307ec1cb3',  # noqa: E501
    'coverage',
    'flake8',
    'flake8-docstrings==1.3.0',
    'flake8-import-order',
    'nose2',
    'pep8-naming==0.7.0',
    'pre-commit',
    'pylint==1.9.2',
    'sphinx',
    'sphinx_rtd_theme',
    'yapf',
]

with open('README.md') as f:
    readme = f.read()

# Get the package version dynamically
exec(open('./garage/__version__.py').read())  # pylint: disable=exec-used
version = __version__  # noqa: F821, pylint: disable=undefined-variable

setup(
    name='rlgarage',
    version=version,
    author='Reinforcement Learning Working Group',
    description='A framework for reproducible reinforcement learning research',
    url='https://github.com/rlworkgroup/garage',
    packages=[
        package for package in find_packages() if package.startswith('garage')
    ],
    scripts=['scripts/garage'],
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
