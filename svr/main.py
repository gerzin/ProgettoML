#!/usr/bin/env python3
from myutils import *
from validation_utils import *
import numpy as np
from SVR import SVR as svr1
from sklearn.svm import SVR as svr2



if __name__ == '__main__':
    args = get_cmdline_args()
    X, Y1, Y2 = load_data(args.file)
    
    _C = 30
    _eps = 1
    folds = k_fold_indeces(len(X), 3)
    tr, tr_y, vs, vs_y = split_dataset(X, Y1, folds[0], folds[1])
    
    reg1 = svr2(C=_C, epsilon=_eps)
    reg1.fit(tr, tr_y)
    o1 = reg1.predict(vs)
    print(reg1.intercept_)
    
    g = 1/(len(X[0])*tr.var())
    reg2 = SVR(gamma=g, C=_C, eps=_eps, tol=0.001, maxIter=5000)
    reg2.fit(tr, tr_y)
    #reg2.bias = reg1.intercept_[0]
    o2 = [ reg2.predict(x) for x in vs ]
    print(reg2.bias)
    
    print(f"error1: {np.square(o1 - vs_y).mean()}")
    print(f"error2: {np.square(o2 - vs_y).mean()}")