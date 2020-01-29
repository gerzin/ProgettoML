#!/usr/bin/env python3
"""
Compute and print the results for the ML-CUP.
Train both the SVR on the training set, predict 
the values on the test set and print the results on a file.
"""
import numpy as np
from myutils import *
from SVR import SVR
import sys
import csv

if __name__ == '__main__':
    args = get_cmdline_args("")
    print("loading data...")
    trX, trY1, trY2 = load_data(args.file)
    if args.file2 is None:
        print(f"usage {sys.argv[0]} -f trainingset.csv -f2 testset.csv")
        sys.exit(1)
    tsX = load_data(args.file2, split=False)
    tol = 1e-3
    maxIter = 5000
    param_reg1 = (0.09, 20, 0.01)
    param_reg2 = (0.15, 10, 0.05)
    
    gamma1, C1, eps1 = param_reg1
    reg1 = SVR(gamma=gamma1, C=C1, eps=eps1, tol=tol, maxIter=maxIter)
    
    gamma2, C2, eps2 = param_reg2
    reg2 = SVR(gamma=gamma2, C=C2, eps=eps2, tol=tol, maxIter=maxIter)
    print("training regressor for column 1...")
    reg1.fit(trX, trY1)
    print("training regressor for column 2...")
    reg2.fit(trX, trY2)
    print("computing the outputs of the regressors...")
    output_reg1 = [ reg1.predict(x) for x in tsX ]
    output_reg2 = [ reg2.predict(x) for x in tsX ]
    
    filename = "blind_test_results.csv"

    print(f"writing results on f{filename}")
    with open(filename, "w") as f:
        f.write("#Gerardo Zinno / Marco Lepri\n# placeholderName\n# ML-CUP19V1\n #29/01/2020\n")
        writer = csv.writer(f)    
        for (n, (o1,o2)) in enumerate(zip(output_reg1, output_reg2)):
            print(f"{n+1}\t{o1}\t{o2}")
            writer.writerow([n+1, o1, o2])

