.. _installation:


============
Installation
============

Install Garage in a Python Environment
--------------------------------------

Garage is a Python package which can be installed in most Python 3.5+ environments using standard commands, i.e.

.. code-block:: bash

    pip install --user garage


We recommend you build your project using a Python environment manager which supports dependency resolution, such as `pipenv <https://docs.pipenv.org/en/latest/>`_, `conda <https://docs.conda.io/en/latest/>`_, or `poetry <https://poetry.eustace.io/>`_. We test against pipenv and conda.

Garage is also tested using `virtualenv <https://virtualenv.pypa.io/en/latest/>`_. However, virtualenv has difficulty resolving dependency conflicts which may arise between garage and other packages in your project, so additional care is needed when using it. You are of course free to install garage as a system-wide Python package using pip, but we don't recommend doing so.

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


Install Environment Dependencies (Optional)
-------------------------------------------

Generally speaking, system dependencies of garage are minimal, and likely already installed.
However, many of the environments garage is used with have additional
dependencies, and we provide "setup scripts" for installing those dependencies
and working around known problems on common platforms.

If you can already use the environments you need, you can skip this section.

A MuJoCo key is required to run these install scripts. You can get one here: https://www.roboti.us/license.html

In order to use those scripts, please do the following:

Clone our repository (https://github.com/rlworkgroup/garage) and navigate to its directory.

Then, from the root directory of the repo, run the script.

- On Linux:

.. code-block:: bash

    ./scripts/setup_linux.sh --mjkey path-to-your-mjkey.txt

- On macOS:

.. code-block:: bash

    ./scripts/setup_macos.sh --mjkey path-to-your-mjkey.txt

If all of the system dependencies were installed correctly, then the exact
version of common RL environments that work with garage can be installed via
pip:

.. code-block:: bash

    pip install 'garage[mujoco,dm_control]'

Extra Steps for Garage Developers
---------------------------------

If you plan on developing the garage repository, as opposed to simply using it as a library, you will probably prefer to install your copy of the garage repository as an editable library instead. After installing the pre-requisites using the instructions in `Install Environment Dependencies (Optional)`_, you should install garage in your environment as below.
If you would like to contribute changes back to garage, please also read :code:`CONTRIBUTING.md`.

- pipenv

.. code-block:: bash

    cd path/to/garage/repo
    pipenv --three
    pipenv install --pre -e '.[all,dev]'


- conda

.. code-block:: bash

    conda activate myenv
    pip uninstall garage  # To ensure no existing install gets in the way.
    cd path/to/garage/repo
    pip install -e '.[all,dev]'


- virtualenv

.. code-block:: bash

    source myenv/bin/activate
    pip uninstall garage  # To ensure no existing install gets in the way.
    cd path/to/garage/repo
    pip install -e '.[all,dev]'
