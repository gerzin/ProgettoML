# Data directory
this directory contains various csv files constituing the input or the output for the project.

# Files description
* ML-CUP19-TR.csv contains all the available `<input, y1, y2>` pairs.
and has been split into:
    * training.csv containing our training set. (80% of ML-CUP19-TR.csv)
    * test.csv containing our test set. (20% of ML-CUP19-TR.csv)
* GZML_ML-CUP19-TS.csv contains a series of input points of which we have to predict the `<y1, y2>` for the blind test.
* teamname-ML-CUP19-TS.csv contains the results of the blind test.
* Mod_Sel_1(2)_results_1.csv contains the results of the first grid search for the first (second) output of the input pattern.
* Mod_Sel_1(2)_results_2.csv contains the results of the second grid search for the first (second) output of the input pattern. 
* trainingError1(2)_1(2).csv contains the training error for the first (second) column in the first (second) grid search.
