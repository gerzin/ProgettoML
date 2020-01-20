#!/usr/bin/env python3
from myutils import *
import numpy as np
from SLBQP import *
from SVR import SVR
import pickle




if __name__ == '__main__':
    args = get_cmdline_args()
    X, Y1, Y2 = load_data(args.file) 
    print("data loaded...")   
    regressor = SVR(maxIter=3000, eps=1e-4)
    print("start fitting...")

    Yscaled, M, m = scale(Y2)
    if args.percentage is not None:
    	X, V = splitHorizontally(X, args.percentage)
    	Y, YV = splitHorizontally(Yscaled, args.percentage)

    regressor.fit(X, Y)
