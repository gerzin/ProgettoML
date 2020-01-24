#!/usr/bin/env python3
import numpy as np
from numba import njit
import sys
from myutils import *
from inspect import signature
from SVR import *
import itertools as it
from model_selection import *
import math

def random_sampling(arr, n)

class GridSearcher:
    def __init__(self, reg, ranges=None):
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
        print("starting grid search...")
        ind = k_fold_split_indices(X,Y,k)
        #treshold
        th = math.inf
        for i in self.grid:
            gamma, C, eps, tol, maxIter = i
            err = k_fold_evaluate(X, Y, k, ind, gamma, C, eps, maxIter, threshold=th)
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
    gamma = [0.01 , 0.05, 0.1, 0.5, 1., 5.]
    C = [0.05, 0.1, 1., 10., 20., 30.]
    eps = [0.001, 0.005, 0.01, 0.1, 0.5, 1]
    tol = [1e-3]
    maxiter = [5000]
    ranges = (gamma, C, eps, tol, maxiter)
    
    gs = GridSearcher(SVR, ranges)
    gs.start_search(X,Y,args.kfold)

