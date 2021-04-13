#!/usr/bin/env python3
from SLBQP import SLBQP
from myutils import *
from genBCQP import genBCQP
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from time import time
from random import randint
import sys

if __name__ == '__main__':
    n = 500
    u = 1.
    l = -1.
    epsilon = 0.1

    a = randint(1, 10000000)
    a = 6530600
    print(a)
    #Q, q = genBCQP(n, actv=0.9, ecc=0.5, u=u, seed=a)

    feat, targ = load_ml_dataset()
    feat = feat.to_numpy()
    targ = targ.to_numpy()

    Q, q = sample_transform_problem(feat, targ[:, 0], n, a)

    # A = np.block([[Q, -Q], [-Q, Q]])
    # b = np.block([epsilon - q, epsilon + q])
    # a = np.block([np.ones(n), -np.ones(n)])

    
    start1 = time()
    s1, x1, v1, it1 = SLBQP(Q, q, u, epsilon, eps=1e-6, maxIter=-1,
                       lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=False, prj_type=2)
    end1 = time()
    print("info ROS:")
    print(f"x: {x1} ({s1})\tv: {v1}\ttime: {end1-start1}\titer: {it1}")
    

    ######
    start2 = time()
    s2, x2, v2, it2 = SLBQP(Q, q, u, epsilon, eps=1e-6, maxIter=-1,
                       lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=False, prj_type=1)
    end2 = time()
    print("info GOLD:")
    print(f"x: {x2} ({s2})\tv: {v2}\ttime: {end2-start2}\titer: {it2}")
    ######

    # _G, _A, _h, _b = build_problem(2*n, u)
    # Q1 = matrix(A)
    # q1 = matrix(b)
    # G1 = matrix(_G)
    # h1 = matrix(_h)
    # A1 = matrix(_A)
    # b1 = matrix(_b)
    # start5 = time()
    # sol = solvers.qp(Q1, q1, G1, h1, A1, b1)
    # end5 = time()
    # x5 = np.array(sol['x']).flatten()
    # v5 = sol['primal objective']
    # print("Solver info:")
    # print(f"x: {x5}\tv: {v5}\ttime: {end5-start5}")
