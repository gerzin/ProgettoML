#!/usr/bin/env python3
from SLBQP import SLBQP
from myutils import *
from genBCQP import genBCQP
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

if __name__ == '__main__':    
    n = 200
    u = 1.
    Q, q, a = genBCQP(n, rank=1.1, ecc=0.99, u=u)
    
    
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
    s, sol, v = SLBQP(Q, q, u, a, x, eps=1e-3, maxIter=-1, lmb0=0, d_lmb=2, prj_eps=1e-6, verbose=True)
    print(f"{v1} \t {v} ({s})")
    print(np.linalg.norm(x1-sol))
        