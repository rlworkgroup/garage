.. _installation:


============
Installation
============

Express Install
===============

Install System Dependencies
---------------------------

The fastest way to set up dependencies for garage is via running the setup script.

Clone our repository (https://github.com/rlworkgroup/garage) and navigate to its directory.

A MuJoCo key is required for installation. You can get one here: https://www.roboti.us/license.html

Make sure you run these scripts from the root directory of the repo, not from the scripts directory.

- On Linux, run the following:

.. code-block:: bash

    ./scripts/setup_linux.sh --mjkey path-to-your-mjkey.txt --modify-bashrc

- On macOS, run the following:

.. code-block:: bash

    ./scripts/setup_macos.sh --mjkey path-to-your-mjkey.txt --modify-bashrc

Install Garage in a Python Environment
--------------------------------------

The script sets up pre-requisites for each platform, but does not install the Python package. We recommend you build your project using a Python environment manager which supports dependency resolution, such as `pipenv <https://docs.pipenv.org/en/latest/>`_, `conda <https://docs.conda.io/en/latest/>`_, or `poetry <https://poetry.eustace.io/>`_. We test against `pipenv` and `conda`.

garage is also tested using `virtualenv <https://virtualenv.pypa.io/en/latest/>`_, but we recommend against building your project using `virtualenv`, because it has difficulty resolving dependency conflicts which may arise between garage and other packages in your project. You are of course free to install garage as a system-wide Python package using `pip`, but we don't recommend this for the same reasons we recommend against using `virtualenv`.

NOTE: garage only supports Python 3.5+, so make sure you Python environment is using this or a later version.

- pipenv

.. code-block:: bash

    pipenv --three  # garage only supports Python 3.5+
    pipenv install --pre garage  # --pre required because garage has some dependencies with verion numbers <1.0


- conda (environment named "myenv")

.. code-block:: bash

    conda activate myenv
    pip install garage

Alternatively, you can add garage in the pip section of your `environment.yml`

.. code-block:: yaml

    name: myenv
    channels:
      - conda-forge
    dependencies:
    - python>=3.5
    - pip
    - pip
      - garage

- virtualenv (environment named "myenv")

.. code-block:: bash

    source myenv/bin/activate
    pip install garage


Extra Steps for Developers
--------------------------

If you plan on developing the garage repository, as opposed to simply using it as a library, you will probably prefer to install your copy of the garage repository as an editable library instead. After installing the pre-requisites using the instructions in `Install System Dependencies`_, you should install garage in your environment as below.

- pipenv

.. code-block:: bash

    cd path/to/garage/repo
    pipenv --three
    pipenv install --pre -e .[all,dev]


- conda

.. code-block:: bash

    conda activate myenv
    cd path/to/garage/repo
    pip install -e .[all,dev]


- virtualenv

.. code-block:: bash

    source myenv/bin/activate
    cd path/to/garage/repo
    pip install -e .[all,dev]


GPU Support
===========

To enable GPU support, install the `garage[gpu]` extra package into your Python environment.

Before you run garage, you need to specify the directory for the CUDA library in environment variable :code:`LD_LIBRARY_PATH`. You may need to replace the directory conforming to your CUDA version accordingly. We recommend you add this to your shell profile (e.g. `~/.bashrc`) for convenience.

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64


You should now be able to use your GPU with TensorFlow and PyTorch.
