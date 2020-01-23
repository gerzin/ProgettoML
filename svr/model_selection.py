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

def k_fold_evaluate(X, Y, k, ind, gamma, C, eps, maxIter):
    err = 0
    for i in range(k):
        print("start iteration")
        reg = SVR(gamma=gamma, C=C, eps=eps, maxIter=maxIter)        
        trs, trs_targ, vals, vals_targ = test_validation_split(X,Y,ind[i], ind[i+1])
        reg.fit(trs, trs_targ)
        err += reg.evaluateMSE(vals, vals_targ)
        print(f"iteration {i} completed, err: {err}")
    err = err/k
    dump_svr_params("test.csv", (gamma, C, eps, maxIter, err))
    return err

if __name__ == '__main__':
    args = get_cmdline_args()
    print("loading data...")
    X, Y1, Y2 = load_data(args.file)
    Y, M, m = None, None, None
    
    if args.column == 1:
        Y = Y1
    else:
        Y = Y2

    if args.scale:
        print("scaling data...")
        Y, M, m = scale(Y)

    print(f"{args.kfold}-folding...")
    ind = k_fold_split_indices(X, Y, args.kfold)
    
    print(k_fold_evaluate(X, Y, args.kfold, ind, 1/20, 12, 0.05, 5000))

    sys.exit()
    regressor = SVR(maxIter=3000, eps=1e-4)
    print("start fitting...")
    regressor.fit(X, Y)
