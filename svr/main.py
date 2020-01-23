#!/usr/bin/env python3
from myutils import *
import numpy as np
from SVR import SVR as svr1
from sklearn.svm import SVR as svr2



if __name__ == '__main__':
    args = get_cmdline_args()
    X, Y1, Y2 = load_data(args.file) 
    reg = svr2()
    reg.fit(X,Y1)
    indeces = reg.support_
    s1 = reg.dual_coef_
    
    
    g = 1/(len(X[0])*X.var())
    r = svr1(gamma=g, maxIter=5000)
    r.fit(X,Y1)
    
    v = ((Y1 - Y1.mean())**2).sum()
    YP = [r.predict(i) for i in X]
    u = ((Y1 - YP)**2).sum()
    s2 = r.gammas
    
    i = 0
    for j in indeces:
        print(f"{s1[0][i]} = {s2[j]}")
        i += 1
        
    print(1 - u/v)
    print(reg.score(X,Y1))
