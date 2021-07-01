#!/usr/bin/env python3
import time
import sys
import numpy as np
from random import randint

from SLBQP import SLBQP
from myutils import *


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print(f"Use: {sys.argv[0]} n C eps [dataset]")
        print(f"\tdataset = [ml, airfoil, california]")
        exit()


    # Problem parameters
    n = int(sys.argv[1])
    C = float(sys.argv[2])
    epsilon = float(sys.argv[3])

    feat, targ = None, None
    if len(sys.argv) > 4:
        if sys.argv[4].lower().startswith('m'):
            feat, targ = load_ml_dataset()
            feat = feat[:, 0]

        elif sys.argv[4].lower().startswith('a'):
            feat, targ = load_airfoil_dataset()

        elif sys.argv[4].lower().startswith('c'):
            feat, targ = load_california_dataset()

        else:
            print("Unknown dataset... Using Airfoil")
            feat, targ = load_airfoil_dataset()
            targ = targ.flatten()

    else:
        feat, targ = load_airfoil_dataset()
        targ = targ.flatten()
    # ------------------

    # Parameters check
    if n <= 0:
        print(f"Bad problem size: {n=}")
        exit()

    if C <= 0:
        print(f"Bad problem parameter: {C=}")
        exit()

    if epsilon <= 0:
        print(f"Bad problem parameter: {epsilon=}")
        exit()
    # ----------------


    GOLDSTEIN = 1
    ROSEN = 2

    seed = randint(1, 10000000)
    print(f"{seed=}")
    np.random.seed(seed)

    # Generate problem
    Q, q = sample_transform_problem(feat, targ, n)


    ######
    print("Running ROSEN")
    startr = time.time()
    sr, xr, vr, itr = SLBQP(Q, q, C, epsilon, eps=1e-3, maxIter=-1, prj_eps=1e-9, verbose=True, prj_type=ROSEN)
    endr = time.time()
    print("info ROSEN:")
    print(f"x: {xr} ({sr})\tv: {vr}\ttime: {endr-startr}\titer: {itr}\n")
    ######


    ######
    print("Running GOLDSTEIN")
    startg = time.time()
    sg, xg, vg, itg = SLBQP(Q, q, C, epsilon, eps=1e-3, maxIter=-1, prj_eps=1e-9, verbose=True, prj_type=GOLDSTEIN)
    endg = time.time()
    print("info GOLDSTEIN:")
    print(f"x: {xg} ({sg})\tv: {vg}\ttime: {endg-startg}\titer: {itg}\n")
    ######


    ######
    print("Running SOLVER")
    sol = solve_with_cvxopt(Q, q, epsilon, C)
    xo = np.array(sol['x']).flatten()
    vo = sol['dual objective']
    print("info SOLVER:")
    print(f"x: {xo} \tv: {vo}\n")
    ######


    rosen_x_gap = np.linalg.norm(xo - xr) / np.linalg.norm(xo)
    goldstein_x_gap = np.linalg.norm(xo - xg) / np.linalg.norm(xo)

    print(f"\nRosen's solution gap: {rosen_x_gap}")
    print(f"Goldstein's solution gap: {goldstein_x_gap}")


    rosen_v_gap = (vo - vr) / np.abs(vo)
    goldstein_v_gap = (vo - vg) / np.abs(vo)

    print(f"\nRosen's final relative gap: {rosen_v_gap}")
    print(f"Goldstein's final relative gap: {goldstein_v_gap}")


    a = np.block([np.ones(n), -np.ones(n)])

    print(f"\nSolver equality constraint precision: {a @ xo}")
    print(f"Rosen equality constraint precision: {a @ xr}")
    print(f"Goldstein equality constraint precision: {a @ xg}\n")
