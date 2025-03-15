# Precomputed data files 

These files contain intermediate results which take important compute time (in some cases multiple days).
The computational notebooks will check for these; if they find them, computations are skipped and loaded from disk instead.
For data files to be found, they must be at the expected location; if for example the path to the neural circuit notebook is 

    /path/to/notebooks/Ex_Prinz2004.py

then the path to one of the data files should be

    /path/to/notebooks/data/criteria-model-compare

The compressed tar file `tracked_tasks` should be unpack so that there is a folder

    /path/to/notebooks/data/Calibrate

(The data folder can be changed by adding a config file; see instructions in `/path/to/notebooks/config/defaults.cfg`.)

The files under `Calibrate` are [SumatraTask](https://sumatratask.readthedocs.io/) *task files*.
The rest of the files are standard Python shelves.