# SLBQP

#### Folder Structure
* `SLBQP.py`
    * This is the file containing the main algorithm.
* `SLBQP_test.py`
    * This is the file used to test the algorithms.
* `projections.py`
    * This file contains Rosen's and Goldstein's projection functions.
* `./data`
    * This folder contains the ML dataset and the Airfoil dataset. The California Housing dataset, being included is sklearn, is automatically downloaded from the internet when the respective load function is called.
* `./experiments`
    * This folder contains the notebooks we used to run the experiments, generate and analyse the data and create the plots.
        * DataGeneration is the notebook containing the code to run the experiments.
        * ResultsAnalysis is the notebook to load and analyse the data generated with DataGeneration and to show the plots.
* `./myutils`
    * This folder is a package containing utility functions for data loading, data transformation and visualization

