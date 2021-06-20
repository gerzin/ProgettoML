#!/usr/bin/env python3
import numpy as np
from time import time
from random import randint
import sys
from SLBQP import SLBQP
from myutils import *
from cvxopt import matrix, solvers


if __name__ == '__main__':
    n = 600
    u = 0.1
    epsilon = 0.1

    feat, targ = load_airfoil_dataset()

    while True:

        a = randint(1, 10000000)
        a = 4719720
        print(a)
        np.random.seed(a)

        Q, q = sample_transform_problem(feat, targ, n, a)

        ######
        #start1 = time()
        s1, x1, v1, it1 = SLBQP(Q, q, u, epsilon, eps=1e-3, maxIter=-1,
                           lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=True, prj_type=2)
        #end1 = time()
        #print("info ROS:")
        #print(f"x: {x1} ({s1})\tv: {v1}\ttime: {end1-start1}\titer: {it1}")
        ######
        
        ######
        #start2 = time()
        #s2, x2, v2, it2 = SLBQP(Q, q, u, epsilon, eps=1e-3, maxIter=-1,
        #                   lmb0=0, d_lmb=2, prj_eps=1e-9, verbose=False, prj_type=1)
        #end2 = time()
        #print("info GOLD:")
        #print(f"x: {x2} ({s2})\tv: {v2}\ttime: {end2-start2}\titer: {it2}")
        ######

        solvers.options['abstol'] = 1e-12
        solvers.options['reltol'] = 1e-12
        solvers.options['feastol'] = 1e-12

        sol = solve_with_cvxopt(Q, q, epsilon, u)

        x3 = np.array(sol['x']).flatten()
        v3 = sol['dual objective']

        #print(np.linalg.norm(x3 - x1) / np.linalg.norm(x3))
        #print(np.linalg.norm(x3 - x2) / np.linalg.norm(x3))

        #a = np.block([np.ones(n), -np.ones(n)])

        #print(a @ x1)
        #print(a @ x2)
        #print(a @ x3)

        '''
        print("-------")

        for i, (a,b,c) in enumerate(zip(x1,x2,x3)):
            if a > 1e-12 and a < 0.09999 and \
               b > 1e-12 and b < 0.09999 and \
               c > 1e-12 and c < 0.09999:

                if (abs(c-a) > 1e-7 or abs(c-b) > 1e-7):
                    print(i, a, b, c)

            elif not (a < 1e-12 and b < 1e-12 and c < 1e-12) and \
                 not (a > 0.09999 and b > 0.09999 and c > 0.09999):
                
                if (abs(c-a) > 1e-7 or abs(c-b) > 1e-7):
                    print(i, a, b, c)

        print("-------")
        '''


        if v3 >= v1:
            print(v3, v1)
            break