.. garage documentation master file, created by
   sphinx-quickstart on Mon Feb 15 20:07:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to garage
=================

garage is a framework for developing and evaluating reinforcement learning algorithms.

garage is a work in progress, input is welcome. The available documentation is limited for now.

User Guide
==========

The garage user guide explains how to install garage, how to run experiments, and how to implement new MDPs and new algorithms.

.. toctree::
   :maxdepth: 2

   user/installation
   user/experiments
   user/implement_env
   user/implement_algo_basic
   user/implement_algo_advanced


Citing garage
=============

If you use garage for academic research, please cite the repository using the following BibTeX entry. You should update the `commit` field with the commit or release tag your publication uses.


.. code-block:: text

    @misc{garage,
      author = {The garage contributors},
      title = {Garage: A toolkit for reproducible reinforcement learning research},
      year = {2019},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/rlworkgroup/garage}},
      commit = {ebd7800430b0212c3ffcf78fd3ec26b22097c371}
    }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
