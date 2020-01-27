#!/usr/bin/env python3
from SLBQP import SLBQP
from myutils import *
import numpy as np
#from cvxopt import matrix
#from cvxopt import solvers

if __name__ == '__main__':
    args = get_cmdline_args()
    X, y, _ = load_data(args.file, False)

    n = int(len(y)/2)
    a = np.empty(2*n)
    a[:n] = 1.
    a[n:] = -1.
    
#    G, A, h, b = build_problem(len(q), 10)
#    Q1 = matrix(Q)
#    q1 = matrix(q)
#    G1 = matrix(G)
#    h1 = matrix(h)
#    A1 = matrix(A)
#    b1 = matrix(b)
#    sol1 = solvers.qp(Q1,q1,G1,h1,A1,b1)
#    sol1 = np.array(sol1['x']).flatten()
    
    x = np.full(len(y), 5)
    _, sol, v = SLBQP(X, y, 10, a, x, verbose=True)
        