#!/usr/bin/env python3

import numpy as np
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

#@jit(nopython=True, parallel=True)
def compute_x_r(d1, d2, lmb, u):
    """Compute the optimal value for x given lambda.
    Params:
        d1  -- first block of the direction vector
        d2  -- second block of the direction vector
        lmb -- current value for lambda
        u   -- upper bound of the feasible region
    Returns:
        x1  -- first block of the optimal x
        x2  -- second block of the optimal x
        r   -- value of r given the optimal x and lambda
    """
    
    # x = d + lmb * a
    # with a = [1...1,-1...-1]
    x1 = d1 + lmb
    x2 = d2 - lmb

    # 'Apply' the box constraints
    x1 = np.clip(x1, 0, u)
    x2 = np.clip(x2, 0, u)
       
    # r = a'x
    # with a = [1...1,-1...-1]
    r = np.sum(x1) - np.sum(x2)
    
    return x1, x2, r

#@jit(nopython=True)
def project_Goldstein(d1, d2, u, lmb, d_lmb, eps):
    """ Return the projection of the point d over the feasible region
    defined by 0 <= x <= u and ax = 0 with a = [1...1,-1...-1]

    Params:
        d1      -- first block of the direction vector
        d2      -- second block of the direction vector
        u       -- upper bound of the feasible region
        lmb     -- 
        d_lmb   -- delta lambda
        eps     -- precision

    Returns:
        x1      -- first block of the projected point
        x2      -- second block of the projected point
    """

    # BRACKETING PHASE -----
    # ----------------------
    
    # Compute x and r and check whether it found the minimum
    x1, x2, r = compute_x_r(d1, d2, lmb, u)
    if abs(r) < eps:
        return x1, x2
    
    if r < 0:
        # r(λ) < 0 -> search for a λ | r(λ) > 0
                
        # initialize lower bounds and update
        lmb_l = lmb;
        r_l = r;
        lmb += d_lmb
        x1, x2, r = compute_x_r(d1, d2, lmb, u)
        if abs(r) < eps:
            return x1, x2
        
        while r < 0:
            # update lower bounds and lambda
            lmb_l = lmb
            r_l = r
            s = max(r_l/r -1, 0.1); d_lmb += d_lmb/s; lmb += d_lmb
            
            # Compute x and r and check whether it found the minimum
            x1, x2, r = compute_x_r(d1, d2, lmb, u)
            if abs(r) < eps:
                return x1, x2
        
        # initialize upper bounds
        lmb_u = lmb
        r_u = r

    else:
        # r(λ) > 0 -> search for a λ' | r(λ') < 0
        
        # initialize upper bounds and update lambda
        lmb_u = lmb
        r_u = r
        lmb -= d_lmb
        
        # Compute x and r and check whether it found the minimum
        x1, x2, r = compute_x_r(d1, d2, lmb, u)
        if abs(r) < eps:
            return x1, x2
        
        while r > 0:
            # update upper bounds and lambda
            lmb_u = lmb; r_u = r
            s = max(r_u/r -1, 0.1)
            d_lmb += d_lmb/s
            lmb -= d_lmb
            
            # Compute x and r and check whether it found the minimum
            x1, x2, r = compute_x_r(d1, d2, lmb, u)
            if abs(r) < eps:
                return x1, x2
        
        # initialize lower bounds
        lmb_l = lmb; r_l = r
    
    #secant phase
    s = 1 - r_l/r_u
    d_lmb = d_lmb/s
    lmb = lmb_u - d_lmb
    x1, x2, r = compute_x_r(d1, d2, lmb, u)

    while(abs(r) >= eps):
        if(r > 0):
            # move upper bound
            if(s <= 2):
                lmb_u = lmb; r_u = r
                s = 1 - r_l/r_u; d_lmb = (lmb_u - lmb_l)/s
                lmb = lmb_u - d_lmb
            else:
                s = max(r_u/r -1, 0.1); d_lmb = (lmb_u - lmb)/s
                lmb_new = max(lmb - d_lmb, 0.75*lmb_l + 0.25*lmb)
                lmb_u = lmb; r_u = r; lmb = lmb_new
                s = (lmb_u - lmb_l)/(lmb_u - lmb)
        else:
            # move lower bound
            if(s >= 2):
                lmb_l = lmb; r_l = r
                s = 1 - r_l/r_u; d_lmb = (lmb_u - lmb_l)/s
                lmb = lmb_u - d_lmb
            else:
                s = max(r_l/r -1, 0.1); d_lmb = (lmb - lmb_l)/s
                lmb_new = min(lmb + d_lmb, 0.75*lmb_u + 0.25*lmb)
                lmb_l = lmb; r_l = r; lmb = lmb_new
                s = (lmb_u - lmb_l)/(lmb_u - lmb)

        x1, x2, r = compute_x_r(d1, d2, lmb, u)
                
    return x1, x2

#@njit
def project_Rosen(d1, d2, x1, x2, u, n):
    """ Rosen projection of d over the feasible region 0 <= x <= u

    Params:
        d1  -- first block of the direction vector
        d2  -- second block of the direction vector
        x1  -- first block of the iterate
        x2  -- second block of the iterate
        u   -- upper bound of the feasible region
        n   -- dimension of the problem

    Returns:
        proj1  -- first block of the projected gradient
        proj2  -- second block of the projected gradient
    """

    active_indeces1 = [(x1[i] == 0 and d1[i] < 0) or (x1[i] == u and d1[i] > 0) for i in range(n)]
    active_indeces2 = [(x2[i] == 0 and d2[i] < 0) or (x2[i] == u and d2[i] > 0) for i in range(n)]

    free_indeces1 = [not i for i in active_indeces1]
    free_indeces2 = [not i for i in active_indeces2]
    
    changed = True
    while(changed):

        proj1 = np.zeros(n)
        proj2 = np.zeros(n)

        d_sum = sum(d1[free_indeces1]) - sum(d2[free_indeces2])
        den = (2*n - sum(active_indeces1) - sum(active_indeces2))

        if(den == 0):
            return proj1, proj2
        else:
            v = d_sum / den

        proj1[free_indeces1] = d1[free_indeces1] - v
        proj2[free_indeces2] = d2[free_indeces2] + v
    
        changed = False
        for i in range(n):
            if (x1[i] == 0 and proj1[i] < 0) or (x1[i] == u and proj1[i] > 0):
                active_indeces1[i] = True
                changed = True
                break
    
        if (changed): continue

        for i in range(n):
            if (x2[i] == 0 and proj2[i] < 0) or (x2[i] == u and proj2[i] > 0):
                active_indeces2[i] = True
                changed = True
                break
    
    return proj1, proj2


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
    (d1,d2) = K.shape
    if d1 != n or d2 != n:
        raise ValueError("Q has wrong size")

    if not isinstance(C, np.float64):
        C = np.float64(C)

    if epsilon < 0:
        raise ValueError("epsilon must be positive")

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
    # End of initialization - - - - - - -

    if verbose:
        print("Iter.\tFunction val\t||gradient||\t||direction||\tStepsize\tMaxStep")


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
            d1, d2 = project_Goldstein(x1 - g1, x2 - g2, C, lmb0, d_lmb, prj_eps)
            d1 = d1 - x1
            d2 = d2 - x2
        else:
            d1, d2 = project_Rosen(-g1, -g2, x1, x2, C, n)
        
        # Compute the norm of the gradient (g) and of the direction (d)
        g_norm = np.sqrt((g1 @ g1) + (g2 @ g2))
        d_norm = np.sqrt((d1 @ d1) + (d2 @ d2))
        
        # Print stats - - - - - - - - - - -
        if verbose:
            print("%5d\t%1.8e\t%1.8e\t%1.8e" % (i, v, g_norm, d_norm), end="")
        # - - - - - - - - - - - - - - - - -
        
        # Check for termination
        if(d_norm < eps):
            if verbose : print("")
            return ('optimal', np.block([x1, x2]), v)
        if(maxIter > 0 and i >= maxIter):
            if verbose: print("")
            return ('terminated', np.block([x1, x2]), v)
        
        
        # Compute the maximum feasible stepsize - - - - -
        max_alpha = np.Inf
        for j in range(len(d1)):
            if(d1[j] > 0):
                max_alpha = min( max_alpha, (C - x1[j])/d1[j] )
            elif(d1[j] < 0):
                max_alpha = min( max_alpha, (-x1[j])/d1[j] )
        
        for j in range(len(d2)):
            if(d2[j] > 0):
                max_alpha = min( max_alpha, (C - x2[j])/d2[j] )
            elif(d2[j] < 0):
                max_alpha = min( max_alpha, (-x2[j])/d2[j] )
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