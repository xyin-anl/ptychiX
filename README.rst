============
Installation
============

Clone the repository to your workspace, and create a new conda environment
using::

    conda create -n <new_env_name> --file requirements-dev.txt

Then install the package using::

    pip install -e .

=======================
How to run test scripts 
=======================

1. Contact the developers to be given access to the GitLab repository
   that holds test data. 
2. After gaining access, clone the GitLab data repository to your
   hard drive. 
3. Set ``PTYCHO_CI_DATA_DIR`` to the ``ci_data`` directory of the data
   repository: ``export PTYCHO_CI_DATA_DIR="path_to_data_repo/ci_data"``.
4. Run any test scripts in ``tests`` with Python.
