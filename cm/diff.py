#!/usr/bin/env python3
from SLBQP import SLBQP
from myutils import *
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

if __name__ == '__main__':
    args = get_cmdline_args()
    Q, q, _ = load_data(args.file, False)

    n = int(len(q)/2)
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
    x1 = np.array(sol1['x']).flatten()
    v1 = sol1['primal objective']
    
    x = np.full(len(q), 5)
    s, sol, v = SLBQP(Q, q, 10., a, x, eps=1e-6, maxIter=10000, lmb0=0, d_lmb=2, prj_eps=1e-12, verbose=False)
    print(f"{v1} \t {v} ({s})")
    print(np.linalg.norm(x1-sol))
        