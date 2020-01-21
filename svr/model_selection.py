#!/usr/bin/env python3
from myutils import *
import numpy as np
from SLBQP import *
from SVR import SVR
import pickle
import sys
from numba import jit


@jit(nopython=True)
def k_fold_split_indices(X, Y, k):
    """returns indices where to split X and Y"""
    fold_size = int(len(X)/k)
    return [ i*fold_size for i in range(k) ] + [len(X)]

#@jit(nopython=True)
def test_validation_split(X,Y, s, f):
    """Split X, Y into test and validation set"""
    interval = [range(s,f)]
    trs = np.delete(X, interval, 0) # numba doesn't support np.delete
    trs_targ = np.delete(Y, interval, 0)
    vals = X[s:f]
    vals_targ = Y[s:f]
    return trs, trs_targ, vals, vals_targ

def k_fold_evaluate(X, Y, k, ind):
    err = 0
    for i in range(k):
        print("start iteration")
        reg = SVR(maxIter=2000)        
        trs, trs_targ, vals, vals_targ = test_validation_split(X,Y,ind[i], ind[i+1])
        reg.fit(trs, trs_targ)
        err += reg.evaluateMSE(vals, vals_targ)
        print(f"iteration {i} completed, err: {err}")
    
    return err/k

if __name__ == '__main__':
    args = get_cmdline_args()
    print("loading data...")
    X, Y1, Y2 = load_data(args.file)

    print("scaling data...")
    if args.column == 1:
        Yscaled, M, m = scale(Y1)
    else:
        Yscaled, M, m = scale(Y2)
   
    print(f"{args.kfold}-folding...")
    ind = k_fold_split_indices(X, Yscaled, args.kfold)
    
    print(f"X shape = {X.shape}")
    print(k_fold_evaluate(X, Yscaled, args.kfold, ind))

    sys.exit()
    regressor = SVR(maxIter=3000, eps=1e-4)
    print("start fitting...")
    regressor.fit(X, Y)
