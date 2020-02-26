#!/usr/bin/env python3
from SLBQP import SLBQP
from myutils import *
from genBCQP import genBCQP
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from SLBQP2 import SLBQP as TEST
from SLBQP3 import SLBQP as TEST3
from time import time
from random import randint

if __name__ == '__main__':    
    n = 300
    u = 1.
    l = -1.
    epsilon = 0.1
    
    a = randint(1, 10000000)
    print(a)
    Q, q, a = genBCQP(n, actv=0.1, ecc=0.5, u=u, seed=a)
    
    A = np.block([[Q, -Q], [-Q, Q]])
    b = np.block([epsilon - q, epsilon + q])
    a = np.block([np.ones(n), -np.ones(n)])
    
    # Versione ML
    start1 = time()
    s1, x1, v1 = SLBQP(A, b, u, a, eps=1e-6, maxIter=-1, lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=False)
    end1 = time()
    
    # Versione ML con prodotti matrice-vettore "furbi"
#    start2 = time()
#    s2, x2, v2 = TEST3(Q, q, u, epsilon, eps=1e-6, maxIter=-1, lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=True, prj_type=1)
#    end2 = time()
    
    # Versione prof
    start3 = time()
    s3, x3, v3 = TEST3(Q, q, u, epsilon, eps=1e-6, maxIter=5000, lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=True, prj_type=2)
    end3 = time()
    
    print("ML info:")
    print(f"x: {x1} ({s1})\tv: {v1}\ttime: {end1-start1}")
#    
#    print("ML furba info:")
#    print(f"x: {x2} ({s2})\tv: {v2}\ttime: {end2-start2}")
    
    print("Prof info:")
    print(f"x: {x3} ({s3})\tv: {v3}\ttime: {end3-start3}")
    
#    s1, x1, v1 = TEST(Q, q, l, u, verbose=True)
#    print(f"x: {x1} ({s1})")
#    print(v1)
#
#    x = np.zeros(n)
#    s3, x3, v3 = SLBQP(Q, q, u, a, x, eps=1e-6, maxIter=-1, lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=False)
#    print(f"x: {x3} ({s3})")
#    print(v3)
#    
#    G, A, h, b = build_problem(n, u)
#    Q1 = matrix(Q)
#    q1 = matrix(q)
#    G1 = matrix(G)
#    h1 = matrix(h)
#    A1 = matrix(np.array([a]))
#    b1 = matrix(b)
#    start = time()
#    sol1 = solvers.qp(Q1,q1,G1,h1,A1,b1)
#    end = time()
#    print(end-start)
#    x2 = np.array(sol1['x']).flatten()
#    v2 = sol1['primal objective']
#    
#    x = np.zeros(n)
#    start = time()
#    s3, x3, v3 = SLBQP(Q, q, u, a, x, eps=1e-6, maxIter=-1, lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=True)
#    end = time()
#    print(end-start)
#    print(f"{v3} ({s3}) \t {v2} \t {v1} ({s1})")
#    print(np.linalg.norm(x3-x1))