#!/usr/bin/env python3
import numpy as np
from myutils import *
from SVR import *
from model_selection import *
import sys
from math import sqrt

def mean_euclidean_error(out, Y1, Y2):
    err = 0
    for i in range(len(out)):
        a, b = out[i]
        diff1 = (Y1[i]-a)**2
        diff2 = (Y2[i]-b)**2
        err += sqrt(diff1+diff2)
    return err/len(out)

if __name__ == '__main__':
    args = get_cmdline_args("Mean Euclidean Error evaluation")
    if args.file2 is None:
        raise "You must pass the test set"
    print("loading data...")
    X, Y1, Y2 = load_data(args.file)
    tol, maxIter = 1e-3, 5000
    gamma1, C1, eps1 = 0.09, 20, 0.01
    gamma2, C2, eps2 = 0.15, 10, 0.05
    regressor1 = SVR(gamma=gamma1, C=C1, eps=eps1, tol=tol, maxIter=maxIter)
    regressor2 = SVR(gamma=gamma2, C=C2, eps=eps2, tol=tol, maxIter=maxIter)
    print("fitting regressor 1° column")
    regressor1.fit(X, Y1)
    print("fitting regressor 2° column")
    regressor2.fit(X, Y2)

    ts, ts_y1, ts_y2 = load_data(args.file2)
    output = []
    for i in ts:
        a = regressor1.predict(i)
        b = regressor2.predict(i)
        output.append((a,b))
    print("computing mee...")
    ts_mee = mean_euclidean_error(output, ts_y1, ts_y2)
    print(f"Mean Euclidean Error on test set =\t {ts_mee}")
    output = []
    for i in X:
        a = regressor1.predict(i)
        b = regressor2.predict(i)
        output.append((a,b))
    tr_mee = mean_euclidean_error(output, Y1, Y2)
    print(f"Mean Euclidean Error on training set =\t {tr_mee}")

    




