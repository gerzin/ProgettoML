#!/usr/bin/env python3
import numpy as np

from myutils import *
from SVR import *
from model_selection import *
import itertools as it
import sys
from validation_utils import k_fold_evaluate_training_error



if __name__ == '__main__':

    args = get_cmdline_args()
    print("loading data...")
    X, Y1, Y2 = load_data(args.file, delfirst=True, shuffle=False, split=True)
    
    k = 3
    gamma = [0.05, 0.1, 0.5]
    C = [10, 20, 30]
    eps = [0.005, 0.01, 0.1]
    tol = [1e-3]
    maxiter = [5000]
    
    combinations = [i for i in it.product(*(gamma, C, eps, tol, maxiter))]
    ind = k_fold_split_indices(X,Y1,k)
    for i in combinations:
        gamma, C, eps, tol, maxiter = i
        err = k_fold_evaluate_training_error(X, Y1, k, ind, [gamma, C, eps, tol, maxiter], filename="trainingError1.csv")
        print(f"{gamma}\t{C}\t{eps}\t{tol}\t{maxiter}\t{err}")
    
    





