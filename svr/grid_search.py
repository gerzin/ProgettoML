#!/usr/bin/env python3
import numpy as np
import sys
from inspect import signature
from SVR import SVR
import itertools as it
from myutils import *
from validation_utils import *
import math

class GridSearcher:
    def __init__(self, reg, ranges=None):
        '''
        Params:
            reg : regressor class
            ranges : tuple containing the ranges of the parameters to "gridsearch"
        '''
        param_names = [p for p in signature(reg.__init__).parameters]
        param_names = param_names[2:]
        self.grid = None
        if len(param_names) != len(ranges):
            raise "Wrong number of ranges"
        self.param_ranges = zip(param_names, ranges)
        self.ranges = ranges
        print(*self.param_ranges)
        for a,b in self.param_ranges:
            print(f"{a} {b}")
        self._generate_grid()
    
    def _generate_grid(self):
        print("generating grid...")
        self.grid = it.product(*self.ranges)
        self.grid = [i for i in self.grid]

    def start_search(self, X, Y, k):
        """Start the gridsearch.
        
        Params:
            X   -- inputs
            Y   -- outputs
            k   -- number of folds for the k-fold
        """
        print("starting grid search...")
        ind = k_fold_indeces(len(X),k)
        #treshold
        th = math.inf
        for i in self.grid:
            gamma, C, eps, tol, maxIter = i
            err = k_fold_evaluate(X, Y, k, ind, [gamma, C, eps, tol, maxIter], threshold=th)
            print(f"{gamma}\t{C}\t{eps}: {err}")
            if 0 <= err < th:
                th = k*err 

if __name__ == '__main__':
    
    args = get_cmdline_args()
    print("loading data...")
    X, Y1, Y2 = load_data(args.file)
    Y, M, m = None, None, None
    if args.kfold is None:
        print("Error: -k [n] needed")
        sys.exit(1)

    if args.column == 1:
        Y = Y1
    else:
        Y = Y2

    if args.scale:
        print("scaling data...")
        Y, M, m = scale(Y) 
    #intervals for grid search
    gamma = [0.09 , 0.1, 0.105, 0.11, 0.15, 0.17]
    C = [9, 10, 10.5, 11, 14, 19]
    eps = np.linspace(0.05, 0.15, 5).tolist()
    tol = [1e-3]
    maxiter = [5000]
    ranges = (gamma, C, eps, tol, maxiter)
    
    gs = GridSearcher(SVR, ranges)
    gs.start_search(X,Y,args.kfold)