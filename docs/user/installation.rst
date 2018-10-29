.. _installation:


============
Installation
============

Express Install
===============

The fastest way to set up dependencies for garage is via running the setup script.

Clone our repo (https://github.com/rlworkgroup/garage) and navigate to its directory.

A MuJoCo key is required for installation. You can get one here: https://www.roboti.us/license.html

Make sure you run these scripts from the root directory of the repo, not from the scripts directory.

- On Linux, run the following:

.. code-block:: bash

    ./scripts/setup_linux.sh --mjkey path-to-your-mjkey.txt --modify-bashrc

- On macOS, run the following:

.. code-block:: bash

    ./scripts/setup_macos.sh --mjkey path-to-your-mjkey.txt --modify-bashrc


The script sets up a conda environment, which is similar to :code:`virtualenv`. To start using it, run the following:

.. code-block:: bash

    source activate garage


Optionally, if you would like to run experiments that depends on the MuJoCo environment, you can set it up by running the following command:

.. code-block:: bash

    ./scripts/setup_mujoco.sh

and follow the instructions. You need to have the zip file for Mujoco v1.50 and the license file ready.



Manual Install
==============

Anaconda
------------

:code:`garage` assumes that you are using Anaconda Python distribution. You can download it from `https://www.continuum.io/downloads<https://www.continuum.io/downloads`.  Make sure to download the installer for Python 2.7.


System dependencies for pygame
------------------------------

A few environments in garage are implemented using Box2D, which uses pygame for visualization.
It requires a few system dependencies to be installed first.

On Linux, run the following:

.. code-block:: bash

  sudo apt-get install swig
  sudo apt-get build-dep python-pygame

On macOS, run the following:

.. code-block:: bash

  brew install swig sdl sdl_image sdl_mixer sdl_ttf portmidi

System dependencies for scipy
-----------------------------

This step is only needed under Linux:

.. code-block:: bash

  sudo apt-get install build-dep python-scipy

Install Python modules
----------------------

.. code-block:: bash

  conda env create -f environment.yml

GPU Support
===========

To enable GPU support, you need to run the express installation script with the argument :code:`--gpu`. This options installs GPU-supported Tensorflow and modules needed by Theano.

Before you run garage, you need to specify the directory for the CUDA library in environment variable :code:`LD_LIBRARY_PATH`. You may need to replace the directory conforming to your CUDA version accordingly.

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64


You should now be able to use GPU in Tensorflow. For Theano, two additional steps are needed.

* Specify CUDA root in :code:`~/.theanorc` (Create the file if it doesn't exist)

.. code-block:: ini

    [cuda]
    root = /usr/local/cuda-9.0

* | `Enable GPU for theano <http://deeplearning.net/software/theano/tutorial/using_gpu.html>`_ by

.. code-block:: bash

    export THEANO_FLAGS=device=cuda,floatX=float32,force_device=True
