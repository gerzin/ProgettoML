#!/usr/bin/env python3
from SLBQP import *
from myutils import *
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from SVR import *

if __name__ == '__main__':
    args = get_cmdline_args()
    X, y, _ = load_data(args.file, False)
    G, A, h, b = build_problem(len(y), 10)
    
    K = compute_rbf_matrix(X, 1)
    (n,m) = K.shape
    Q = np.block([
        [K, -K],
        [-K, K]
    ])
    
    q = np.empty(2*n)
    q[:n] = 0.1 - y
    q[n:] = 0.1 + y
    
    a = np.empty(2*n)
    a[:n] = 1.
    a[n:] = -1.
    
    G, A, h, b = build_problem(len(q), 10)
    
    Q1 = matrix(Q)
    q1 = matrix(q)
    G1 = matrix(G)
    h1 = matrix(h)
    A1 = matrix(A)
    b1 = matrix(b)
    
    sol1 = solvers.qp(Q1,q1,G1,h1,A1,b1)
    sol1 = np.array(sol1['x']).flatten()
    #print(sol1)
    
    x = np.full(len(q), 5)
    _, sol, v = SLBQP(Q, q, 10, A[0], x)
    #print(sol)
    
    for (x,y) in zip(sol,sol1):
        print(abs(x-y))
    print(np.linalg.norm(sol-sol1))
        
    