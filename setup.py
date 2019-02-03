from setuptools import find_packages
from setuptools import setup

# Required dependencies
required = [
    # Please keep alphabetized
    'awscli',
    'boto3',
    'cached_property',
    'cloudpickle',
    'cma==1.1.06',
    # dm_control throws an error during install about not being able to
    # find a build dependency (absl-py). Later pip executes the `install`
    # command again and the install succeeds because absl-py has been
    # installed. This is stupid, but harmless.
    'dm_control @ https://api.github.com/repos/deepmind/dm_control/tarball/92f9913013face0468442cd0964d5973ea2089ea',  # noqa: E501
    'flask',
    'gym[all]==0.10.11',
    'box2d-py>=2.3.4',
    'hyperopt',
    'ipdb',
    'ipywidgets',
    'jsonmerge',
    'joblib<0.13,>=0.12',
    'jupyter',
    'mako',
    'matplotlib',
    'memory_profiler',
    'mujoco-py<1.50.2,>=1.50.1',
    'numpy==1.14.5',
    'pandas',
    'path.py',
    'plotly @ https://api.github.com/repos/plotly/plotly.py/tarball/2594076e29584ede2d09f2aa40a8a195b3f3fc66',  # noqa: E501
    'polling',
    'protobuf',
    'psutil',
    'pygame',
    'pyglet',
    'PyOpenGL',
    'pyprind',
    'python-dateutil',
    'pyzmq',
    'scikit-image',
    'scipy',
    'tensorboard',
    'tensorflow<1.13,>=1.12.0',
]  # noqa: E501

# Dependencies for optional features
extras = {}
extras['all'] = list(set(sum(extras.values(), [])))

# Development dependencies (*not* included in "all")
extras['dev'] = [
    # Please keep alphabetized
    'baselines @ https://api.github.com/repos/openai/baselines/tarball/f2729693253c0ef4d4086231d36e0a4307ec1cb3',  # noqa: E501
    'codacy-coverage',
    'coverage',
    'coveralls',
    'flake8',
    'flake8-docstrings==1.3.0',
    'flake8-import-order',
    'nose2',
    'pep8-naming==0.7.0',
    'pre-commit',
    'pylint==1.9.2',
    'yapf',
]

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='rlgarage',
    version='2019.02dev',
    author='Reinforcement Learning Working Group',
    description='A framework for reproducible reinforcement learning research',
    url='https://github.com/rlworkgroup/garage',
    packages=[
        package for package in find_packages() if package.startswith('garage')
    ],
    install_requires=required,
    extras_require=extras,
    license=license,
    long_description=readme,
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
