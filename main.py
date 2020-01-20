#!/usr/bin/env python3
from myutils import *
import numpy as np
from SLBQP import *
from SVR import SVR




if __name__ == '__main__':
    args = get_cmdline_args()
    X, Y1, Y2 = load_data(args.file) 
    print("data loaded...")   
    regressor = SVR(maxIter=2000)
    print("start fitting...")
    Yscaled, M, m = scale(Y2)
    regressor.fit(X, Yscaled)
