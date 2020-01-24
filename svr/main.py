#!/usr/bin/env python3
from myutils import *
from validation_utils import *
import numpy as np
from SVR import SVR as svr1
from sklearn.svm import SVR as svr2



if __name__ == '__main__':
    args = get_cmdline_args()
    X, Y1, Y2 = load_data(args.file)
    
    folds = k_fold_indeces(len(X), 3)
    tr, tr_y, vs, vs_y = split_dataset(X, Y1, folds[0], folds[1])
    
#    reg1 = svr2()
#    reg1.fit(tr, tr_y)
#    o1 = reg1.predict(vs)
#    print(np.square(o1 - vs_y).mean())
#    
#    g = 1/(len(X[0])*X.var())
#    reg2 = SVR(gamma=g, C=1, eps=0.1, tol=0.001, maxIter=5000)
#    reg2.fit(tr, tr_y)
#    o2 = [ reg2.predict(x) for x in vs ]
#    print(np.square(o2 - vs_y).mean())
    
    
    
    g = 1/(len(X[0])*X.var())
    folds = k_fold_indeces(len(X), 3)
    e = k_fold_evaluate(X, Y1, 3, folds, [0.1, 1, 0.1, 0.001, 5000])
    print(e)
