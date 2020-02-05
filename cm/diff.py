#!/usr/bin/env python3
from SLBQP import SLBQP
from myutils import *
from genBQGP import genBCQP
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

if __name__ == '__main__':
#    args = get_cmdline_args()
#    Q, q, _ = load_data(args.file, False)

#    n = int(len(q)/2)
#    a = np.empty(2*n)
#    a[:n] = 1.
#    a[n:] = -1.
    
    n = 10
    u = 10.
    Q, q, a = genBCQP(n)
    
    
    G, A, h, b = build_problem(n, u)
    Q1 = matrix(Q)
    q1 = matrix(q)
    G1 = matrix(G)
    h1 = matrix(h)
    A1 = matrix(np.array([a]))
    b1 = matrix(b)
    sol1 = solvers.qp(Q1,q1,G1,h1,A1,b1)
    x1 = np.array(sol1['x']).flatten()
    v1 = sol1['primal objective']
    
    x = np.zeros(n)
    s, sol, v = SLBQP(Q, q, u, a, x, eps=1e-6, maxIter=-1, lmb0=0, d_lmb=2, prj_eps=1e-12, verbose=True)
    print(f"{v1} \t {v} ({s})")
    print(np.linalg.norm(x1-sol))
        