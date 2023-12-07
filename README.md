# Notebooks for the EMD paper

## Format

Notebooks are stored in a plain text format using [jupytext](https://jupytext.readthedocs.io/).

## Installation

* Clone this repo:

      git clone git@github.com:alcrene/emd-paper-notebooks.git

* Add the following folders.
  These are used to store computation outputs, and so are not tracked with version control; I like to use [symbolic links](https://linuxhandbook.com/symbolic-link-linux/) for these, to keep them cleanly separated from the tracked files. (This also makes it easier to schedule a separate backup schedule for the data files.)
  This is not required however: using normal directories also works.

  - _data_
    + If you want to reproduce the results from the paper, you may download are results directory from here: (TODO: UPLOAD AND ADD LINK). In particular this will allow you to skip the calibration computations (which can take days on a simple machine), if you leave the parameters unchanged.

  - _figures_
    + If you are also using the [paper repo](https://github.com/alcrene/emd-paper) to rebuild our paper, the produced figures must be in this location:
    
          /path-to-paper-repo/figures

      The easiest way to achieve this is to make _figures_ a symlink to that location, so notebooks automatically place figures in the right directory when they are run. Otherwise, it is also possible of course to run the notebooks and copy the figures directory to the right location afterwards.

* Create a virtual environment for running the notebooks.
  (Note: If you also cloned the paper repo, you can use the same environment for both.)
  You can use your preferred method; two good options are:
  
      mamba create -n emd-paper
      mamba activate emd-paper

  or

      python3 -m venv /path/for/venvs/emd-paper
      source /path/for/venvs/emd-paper/bin/activate

  - Install the requirements
    
        pip install -r requirements.txt
    
  - Add the virtual environment to the kernels available to jupyter.
    
        python -m ipykernel install --user --name emd-paper --display-name "Python (emd-paper)"
    
    We recommend using the same name (`emd-paper`) as above; this way the notebooks will already know which kernel (i.e. environment) to use when you open them. Otherwise you will need to select the correct kernel from the dropdown box.
    This is doubly recommended if you intend to rebuild the paper. (As otherwise each notebook would need to be opened and its kernel updated.)

* Initialize a [SumatraTask](https://sumatratask.readthedocs.io/) project:

      smttask init

## Configuration

- Import the code once: from the root folder, run `import config` in a Python terminal.  
  This will create a *local.cfg* file in the root project directory.
- Open *local.cfg* and edit to your preference.
  - Highly recommended is to activate on-disk cache for *emd_falsify*. With on-disk caching, when a `Calibrate` task is re-executed with different `c` values, only the calculations for the new values are computed.

