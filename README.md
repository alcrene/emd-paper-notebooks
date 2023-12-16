# Code notebooks for the EMD falsification paper

This repository contains the Jupyter notebooks used to create the figures for our paper describing model falsification using the EMD (empirical mean discrepancy).

## Format

Notebooks are stored in a plain text format using [jupytext](https://jupytext.readthedocs.io/).

## Installation

* Clone this repo:

      git clone git@github.com:alcrene/emd-paper-notebooks.git

* Add the following folders.
  These are used to store computation outputs, and so are not tracked with version control; I like to use [symbolic links](https://linuxhandbook.com/symbolic-link-linux/) for these, to keep them cleanly separated from the tracked files. (This also makes it easier to schedule a separate backup schedule for the data files.)
  This is not required however: using normal directories also works.

  - _data_
    + If you want to reproduce the results from the paper, you may download are results directory from here: (TODO: UPLOAD AND ADD LINK). In particular this will allow you to skip the calibration computations (which can take days on a simple machine), if you leave the task parameters unchanged.

  - _figures_
    + If you are also using the [paper repo](https://github.com/alcrene/emd-paper) to rebuild our paper, the produced figures must be in this location:
    
          /path-to-paper-repo/figures

      The easiest way to achieve this is to make _figures_ a symlink to that location, so notebooks automatically place figures in the right directory when they are run. Otherwise, it is also possible of course to run the notebooks and copy the figures directory to the right location afterwards.

* Create a virtual environment and IPython kernel for running the notebooks.
  The recommended and best tested procedure is to use [poetry](https://python-poetry.org/docs/#installation). This repo includes a `poetry.lock` file to make the execution environment fully reproducible.
  However it is also possible to install within any virtual environment using the requirements file.
  
  - **poetry installation**
  
    + Install the dependencies:
  
          poetry install --no-root
        
    + Create an IPython kernel so the environment is accessible to Jupyter notebooks.
      It is easiest to keep the kernel names unchanged, so the match the names saved within the notebooks
      This is doubly recommended if you intend to rebuild the paper. (As otherwise each notebook would need to be opened and its kernel updated.)
    
          poetry shell
          python -m ipykernel install --user --name emd-paper --display-name "Python (emd-paper)"        
          deactivate

  - **alternative installation**
  
    + Create a virtual environment, either with Python’s builtin `venv` 
    
          python3 -m venv /path/for/venvs/emd-paper
          source /path/for/venvs/emd-paper/bin/activate
    
    
      or with mamba/conda:
    
          mamba create -n emd-paper
          mamba activate emd-paper
          
    + Install the dependencies
    
          pip install -r requirements.txt
          
    + Create an IPython kernel so the environment is accessible to Jupyter notebooks.
    
          python -m ipykernel install --user --name emd-paper --display-name "Python (emd-paper)"        
          deactivate  # or `mamba deactivate`

* Initialize a [SumatraTask](https://sumatratask.readthedocs.io/) project:

      smttask init

## Configuration

- Import the code once: from the root folder, run `import config` in a Python terminal.  
  This will create a *local.cfg* file in the root project directory.
- Open *local.cfg* and edit to your preference.
  - Highly recommended is to activate on-disk cache for *emd_falsify*. With on-disk caching, when a `Calibrate` task is re-executed with different `c` values, only the calculations for the new values are computed.

