#!/usr/bin/env python3

import numpy as np
import itertools
from projections import project_Rosen, project_Goldstein
#from numba import jit, njit

"""
SLBQP: Solves the singly linearly box contrained quadratic problem:

    min { (1/2)x'Qx + qx : ax = 0, 0 <= x <= u }

using the projected gradient method.

For the projection, two options are available:

1. Project the point, found by following the minimization direction, onto the feasible region.
    The projection is computed with the algorithm described in [?]:
    
    The projection is the solution of the minimization problem:
        min { (1/2)z'z - d'z : 0 <= z <= u, a'z = 0 }

    It can be solved using the Lagrangian function:
        ϕ(x,λ) = (1/2)z'z - d'z - λa'z
    For a given λ we may solve:
        min { ϕ(x) : 0 <= z <= u }
    This is trivial as the problem separates into n independent subproblems.

    The point is to find a λ such that r(λ) = a'x(λ) = 0 in order to satisfy the KKT
    conditions for the initial minimization problem.
    To do so, we look for the root of r(λ) using the secand method.
    Therefor, we first look for an interval containing the solution (BRACKETING PHASE)
    and then we look for the root (SECANT PHASE)

2. Prohect the gradient so that, moving along its direction, the iteration point remains feasible.
    ...

"""


def SLBQP(K, y, C, epsilon, eps=1e-6, maxIter=1000, lmb0=0, d_lmb=2, prj_eps=1e-6, verbose=False, prj_type=1):
    """
    Solve the quadratic problem using a projected gradient method.

    Params:
        K           -- block of the matrix Q =  [ K -K
                                                 -K  K ]
        y           -- coeff. used to build q (q = [epsilon - y, epsilon + y])
        C           -- upper bound for the feasible points
        epsilon     -- coeff. used to build q (q = [epsilon - y, epsilon + y])
        eps         -- precision for the stopping condition. | direction | < eps
        maxIter     -- max number of iterations (<= 0 for unbounded number of iterations)
        lmb0        -- initial lambda value for the projection algorithm
        d_lmb       -- initial delta_lambda value for the projection algorithm
        prj_eps     -- precision of the projection
        verbose     -- print more stats
        prj_type    -- type of projection. 1 = Goldstein, 2 = Rosen

    Returns:
        status      -- status of the execution. "terminated" if the program
                       has reached MaxIter. "optimal" otherwise.
        x           -- point that minimize the function
        v           -- minimum value of the function
    """

    # Problem dimension
    n = len(y)

    # Input check - - - - - - - - - - -
    (d1, d2) = K.shape
    if d1 != n or d2 != n:
        raise ValueError("Q has wrong size")

    if not isinstance(C, np.float64):
        C = np.float64(C)

    if eps < 0:
        print("eps must be positive... replacing it with 1e-6")
        eps = 1e-6
    # End of input check - - - - - - - - -

    # Initialization - - - - - - - - - - -
    # x = [x1, x2]
    x1 = np.full(n, C/2)
    x2 = np.full(n, C/2)

    # q = [q1, q2]
    q1 = epsilon - y
    q2 = epsilon + y

    i = 0
    # End of initialization - - - - - - -

    if verbose:
        if(prj_type == 1):
            print("Iter.\tFunction val\t||gradient||\t||direction||\tStepsize\tMaxStep")
        else:
            print("Iter.\tFunction val\tDegen.\t||gradient||\t||direction||\tStepsize\tMaxStep")

    while True:
        # Compute function value (v) and gradient (g) - - -
        Kx1 = K @ x1
        Kx2 = K @ x2
        Qx1 = Kx1 - Kx2
        Qx2 = -Kx1 + Kx2

        v = (0.5)*((Qx1 @ x1) + (Qx2 @ x2)) + (q1 @ x1) + (q2 @ x2)

        # g = [g1, g2]
        g1 = Qx1 + q1
        g2 = Qx2 + q2
        # - - - - - - - - - - - - - - - - - - - - - - - - -

        # Compute descent direction
        if prj_type == 1:
            d1, d2 = project_Goldstein(
                x1 - g1, x2 - g2, C, lmb0, d_lmb, prj_eps)
            d1 = d1 - x1
            d2 = d2 - x2
            count = 0
        else:
            d1, d2, count = project_Rosen(-g1, -g2, x1, x2, C)

        # Compute the norm of the gradient (g) and of the direction (d)
        g_norm = np.sqrt((g1 @ g1) + (g2 @ g2))
        d_norm = np.sqrt((d1 @ d1) + (d2 @ d2))

        # Print stats - - - - - - - - - - -
        if verbose:
            print("%5d\t%1.8e\t%5d\t%1.8e\t%1.8e" % (i, v, count, g_norm, d_norm), end="")
        # - - - - - - - - - - - - - - - - -

        # Check for termination
        if(d_norm < eps):
            if verbose:
                print("")
            return ('optimal', np.block([x1, x2]), v)
        if(maxIter > 0 and i >= maxIter):
            if verbose:
                print("")
            return ('terminated', np.block([x1, x2]), v)

        # Compute the maximum feasible stepsize - - - - -
        max_alpha = np.Inf
        for j in range(n):
            if(d1[j] > 0):
                max_alpha = min(max_alpha, (C - x1[j])/d1[j])
            elif(d1[j] < 0):
                max_alpha = min(max_alpha, (-x1[j])/d1[j])

        for j in range(n):
            if(d2[j] > 0):
                max_alpha = min(max_alpha, (C - x2[j])/d2[j])
            elif(d2[j] < 0):
                max_alpha = min(max_alpha, (-x2[j])/d2[j])
        # - - - - - - - - - - - - - - - - - - - - - - - - -

        # Exact line search toward the minimum - - - - -
        # Compute the quadratic part d'Qd
        Kd1 = K @ d1
        Kd2 = K @ d2
        Qd1 = Kd1 - Kd2
        Qd2 = -Kd1 + Kd2
        quad = ((Qd1 @ d1) + (Qd2 @ d2))

        if(quad <= 1e-16):
            # If the quadratic part is zero or negative, take the maximum stepsize
            alpha = max_alpha
        else:
            # Otherwise select the minimum between the optimal unbounded
            # stepsize and the maximum feasible stepsize
            alpha = min(max_alpha, (d_norm**2)/quad)
        # - - - - - - - - - - - - - - - - - - - - - - - -

        # Print stats
        if verbose:
            print("\t%1.8e\t%1.8e" % (alpha, max_alpha))

        # Compute next iterate
        x1 = x1 + alpha * d1
        x2 = x2 + alpha * d2

        i = i + 1
