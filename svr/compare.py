#!/usr/bin/env python3
from myutils import *
from validation_utils import *
import numpy as np
from SVR import SVR as svr1
from sklearn.svm import SVR as svr2



if __name__ == '__main__':
    args = get_cmdline_args()
    X, Y1, Y2 = load_data(args.file)

    Y = None
    if args.column == 1:
        Y = Y1
    else:
        Y = Y2

    folds = k_fold_indeces(len(X), 3)
    tr, tr_y, vs, vs_y = split_dataset(X, Y, folds[0], folds[1])
    
    # Tested parameters (C, eps)
    grid = [(0.1,0.01), (0.1,0.1), (0.1,1),
            (1,0.01), (1,0.1), (1,1),
            (10,0.01),(10,0.1),(10,1),
            (50,0.01), (50,0.1), (50,1)]
    
    # Compute the gamma to match library's one
    g = 1/(len(X[0])*tr.var())
    
    for (C,eps) in grid:
        print(f"C: {C}\teps: {eps}")
        # Library results
        reg1 = svr2(gamma='scale', C=C, epsilon=eps)
        reg1.fit(tr, tr_y)
        o1 = reg1.predict(vs)
        
        # Our results
        reg2 = svr1(gamma=g, C=C, eps=eps, maxIter=10000)
        reg2.fit(tr, tr_y)
        o2 = [ reg2.predict(x) for x in vs ]
    
        print(f"Library error: {np.square(o1 - vs_y).mean()}")
        print(f"Our error: {np.square(o2 - vs_y).mean()}")