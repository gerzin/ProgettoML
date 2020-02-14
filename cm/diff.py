#!/usr/bin/env python3
from SLBQP import SLBQP
from myutils import *
from genBCQP import genBCQP
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from prova import SLBQP as TEST
from time import time

if __name__ == '__main__':    
    n = 4000
    u = 10.
    Q, q, a = genBCQP(n, rank=1.1, ecc=0.99, u=u, seed=42)
    
#    s1, x1, v1 = TEST(Q, q, u, a)
    s1, x1, v1 = 0, 0, 0
    
    G, A, h, b = build_problem(n, u)
    Q1 = matrix(Q)
    q1 = matrix(q)
    G1 = matrix(G)
    h1 = matrix(h)
    A1 = matrix(np.array([a]))
    b1 = matrix(b)
    start = time()
    sol1 = solvers.qp(Q1,q1,G1,h1,A1,b1)
    end = time()
    print(end-start)
    x2 = np.array(sol1['x']).flatten()
    v2 = sol1['primal objective']
    
    x = np.zeros(n)
    start = time()
    s3, x3, v3 = SLBQP(Q, q, u, a, x, eps=1e-6, maxIter=-1, lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=False)
    end = time()
    print(end-start)
    print(f"{v3} ({s3}) \t {v2} \t {v1} ({s1})")
    print(np.linalg.norm(x3-x1))