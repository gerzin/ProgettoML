import numpy as np
from numba import jit, njit, f8, i4

"""
SLBQP: Solves the singly linearly box contrained quadratic problem:

    min { (1/2)x'Qx + qx : ax = 0, 0 <= x <= u }

using the projected gradient method.

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
"""

@jit(nopython=True, parallel=True)
def compute_x(d, lmb, a, u):
    """Compute the optimal value for x given lambda"""
    x = d + lmb*a
    for i in range(len(x)):
        if(x[i] > u): x[i] = u
        elif(x[i] < 0): x[i] = 0
    return x


@jit(nopython=True)
def project(d, u, a, lmb, d_lmb, eps):
    """ Return the projection of the point d over the feasible region
    defined by 0 <= x <= u and ax = 0
    
    """
    # BRACKETING PHASE -----
    # ----------------------
    
    # Compute x and r and check whether it found the minimum
    x = compute_x(d, lmb, a, u); r = np.dot(a, x)
    if(abs(r) < eps): return x
    
    if(r < 0):
        # start looking for a positive value of r
        
        # initialize lower bounds and update lambda
        lmb_l = lmb; r_l = r;
        lmb += d_lmb
        
        # Compute x and r and check whether it found the minimum
        x = compute_x(d, lmb, a, u); r = np.dot(a, x)
        if(abs(r) < eps): return x
        
        while(r < 0):
            # update lower bounds and lambda
            lmb_l = lmb; r_l = r;
            s = max(r_l/r -1, 0.1); d_lmb += d_lmb/s; lmb += d_lmb
            
            # Compute x and r and check whether it found the minimum
            x = compute_x(d, lmb, a, u); r = np.dot(a, x)
            if(abs(r) < eps): return x
        
        # initialize upper bounds
        lmb_u = lmb; r_u = r

    else:
        # start looking for a negative value of r
        
        # initialize upper bounds and update lambda
        lmb_u = lmb; r_u = r; lmb -= d_lmb
        
        # Compute x and r and check whether it found the minimum
        x = compute_x(d, lmb, a, u); r = np.dot(a, x)
        if(abs(r) < eps): return x
        
        while(r > 0):
            # update upper bounds and lambda
            lmb_u = lmb; r_u = r
            s = max(r_u/r -1, 0.1); d_lmb += d_lmb/s; lmb -= d_lmb
            
            # Compute x and r and check whether it found the minimum
            x = compute_x(d, lmb, a, u); r = np.dot(a, x)
            if(abs(r) < eps): return x
        
        # initialize lower bounds
        lmb_l = lmb; r_l = r
    
    #secant phase
    s = 1 - r_l/r_u
    d_lmb = d_lmb/s
    lmb = lmb_u - d_lmb
    x = compute_x(d, lmb, a, u)
    r = np.dot(a, x)
    while(abs(r) >= eps):
        if(r > 0):
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
            if(s >= 2):
                lmb_l = lmb; r_l = r
                s = 1 - r_l/r_u; d_lmb = (lmb_u - lmb_l)/s
                lmb = lmb_u - d_lmb
            else:
                s = max(r_l/r -1, 0.1); d_lmb = (lmb - lmb_l)/s
                lmb_new = min(lmb + d_lmb, 0.75*lmb_u + 0.25*lmb)
                lmb_l = lmb; r_l = r; lmb = lmb_new
                s = (lmb_u - lmb_l)/(lmb_u - lmb)
        x = compute_x(d, lmb, a, u)
        r = np.dot(a, x)
                
    return x

#@jit('numba.float64(numba.array(float64, 2d, C), numba.array(float64, 1d, C), numba.float64, numba.array(float64, 1d, C), numba.array(float64, 1d, C), numba.float64, numba.int64)',nopython=True)
def SLBQP(Q, q, u, a, x, eps=1e-6, maxIter=1000, lmb0=0, d_lmb=2, prj_eps=1e-6, verbose=False): 
    """Solve the quadratic programming problem
            min { (1/2)x'Qx + qx : 0 <= x <= u, a'x = 0}
        using the projected gradient method
    
    Params:
        Q,q     -- factors of the function f(x) = (1/2)x'Qx + qx
        u       -- upper bound for the box constraint 0 <= x <= u
        a       -- weight vector for the linear constraint a'x = 0
        eps     -- precision for the stopping condition (norm of the direction)
        maxIter -- maximum number of iteration
        lmb0    -- initial lambda value for the projection algorithm
        d_lmb   -- initial delta_lambda value for the projection algorithm
        prj_eps -- precision for the stopping condition of the projection algorithm
        verbose -- print algorithm informations at each iteration
    """

    assert len(q)%2 == 0
    n = int(len(q)/2)
    i = 1
    if verbose:
        print("Iter.\tFunction val\t||gradient||\t||direction||\tStepsize")


    while True:
        #if(np.dot(a,x) >= 1e-15):
        #    print(np.dot(a,x))
        
        # Compute function value (v), gradient (g) and descent direction (d)
        Qx = np.dot(Q, x)
        v = (0.5)*np.dot(x,Qx) + np.dot(q, x)
        g = np.array(Qx+q)
        g_norm = np.linalg.norm(g)
        d = np.array(x-g)

        # Project the direction over the feasible region
        d = project(d, u, a, lmb0, d_lmb, prj_eps)
        d = d - x

        # Check for termination
        d_norm = np.linalg.norm(d)
        #print(np.dot(g,d))
        if(d_norm < eps):
            return ('optimal', x, v)
        if(i >= maxIter):
            return ('terminated', x, v)

        # Compute the maximum feasible stepsize
        max_alpha = np.Inf
        for j in range(n):
            if(d[j] > 0):
                max_alpha = min( max_alpha, (u - x[j])/d[j] )
            elif(d[j] < 0):
                max_alpha = min( max_alpha, (-x[j])/d[j] )

        # Exact line search toward the minimum
        quad = np.dot(d, np.dot(Q, d))
        if(quad <= 1e-16):
            # If the quadratic part is zero, take the maximum stepsize
            alpha = max_alpha
        else:
            # Otherwise select the minimum between the optimal unbounded
            # stepsize and the maximum feasible stepsize
            alpha = min(max_alpha, (d_norm**2)/quad)

        if verbose:
            print("%4d\t%1.8e\t%1.8e\t%1.8e\t%1.8e" % (i, v, g_norm, d_norm, alpha))
            input("")

        # Compute next iterate
        x = x + alpha * d

        i = i + 1