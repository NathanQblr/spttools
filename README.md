---
title: README
author: Nathan QUIBLIER
---

![SPToolslogo](img/sptools_logo.svg)

# Synopsis
SPTools is a framework to analyse microscopic single particle trajectories easily and fastly


# Installation

SPTools will be uploaded to the [Python Package Index](https://pypi.org/) once it is officially published. In the meantime, it can be installed from source with any standard Python package manager that supports [pyproject.toml](pyproject.toml) files. For example, to install it with pip, either locally or in a virtual environment, run the following commands:

~~~sh
git clone https://gitlab.inria.fr/nquilbie/sptoolbox
cd sptoolbox
# Uncomment the following 2 lines to create and activate a virtual environment.
# python -m venv venv
# source venv/bin/activate
pip install --upgrade .
~~~




# Usage

There are only two requirements for running an experiment with SPTools:

* A YAML configuration file located at `conf/config_def.yaml` relative to the current working directory.

A different configuration file can be specified by setting the `CONFIG` environment variable. The value of this variable will be interpreted as a subpath within the `conf` directory of the working directory.


Once the configuration file have been created, the experiment can be run with `python3 main.py your_config` (see below).

