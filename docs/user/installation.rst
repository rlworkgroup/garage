.. _installation:


============
Installation
============

Install Garage in a Python Environment
--------------------------------------

Garage is a Python package which can be installed in most Python 3.6+ environments using standard commands, i.e.

.. code-block:: bash

    pip install --user garage


We recommend you build your project using a Python environment manager which supports dependency resolution, such as `pipenv <https://docs.pipenv.org/en/latest/>`_, `conda <https://docs.conda.io/en/latest/>`_, or `poetry <https://poetry.eustace.io/>`_. We test against pipenv and conda.

Garage is also tested using `virtualenv <https://virtualenv.pypa.io/en/latest/>`_. However, virtualenv has difficulty resolving dependency conflicts which may arise between garage and other packages in your project, so additional care is needed when using it. You are of course free to install garage as a system-wide Python package using pip, but we don't recommend doing so.

NOTE: garage only supports Python 3.6+, so make sure you Python environment is using this or a later version. You can still install version 2020.06 if you need to use Python 3.5.

- pipenv

.. code-block:: bash

    pipenv --three  # garage only supports Python 3.6+
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
    - python>=3.6
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

Please note that the setup script for macOS relies on homebrew to install dependencies.
If you don't already have homebrew, the script will attempt to install it. If
you do have homebrew, make sure that your user has permission to install things
with it. Sometimes npm and brew compete for control over folders, and it's easy
to mess up folder permissions when trying to resolve those conflicts. If you
run into these issues, a clean install of homebrew usually solves them.

If all of the system dependencies were installed correctly, then the exact
version of common RL environments that work with garage can be installed via
pip:

.. code-block:: bash

    pip install 'garage[mujoco,dm_control]'

Extra Steps for Garage Developers
---------------------------------

See `here <setting_up_your_development_environment.html#installing-garage-as-an-editable-package>`_

----

This page was authored by K.R. Zentner (`@krzentner <https://github.com/krzentner>`_), with contributions from Gitanshu Sardana (`@gitanshu <https://github.com/gitanshu>`_), Hayden Shively (`@haydenshively <https://github.com/haydenshively>`_), Ryan Julian (`@ryanjulian <https://github.com/ryanjulian>`_), Jonathon Shen (`@jonashen <https://github.com/jonashen>`_), Keren Zhu (`@naeioi <https://github.com/naeioi>`_), Peter Lillian (`@pelillian <https://github.com/pelillian>`_), Gautam Salhotra (`@gautams3 <https://github.com/gautams3>`_), Aman Soni (`@amansoni <https://github.com/amansoni>`_), and Rocky Duan (`@dementrock <https://github.com/dementrock>`_).
