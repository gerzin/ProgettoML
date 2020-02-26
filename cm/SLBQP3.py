import numpy as np
from numba import jit, njit

"""
SLBQP: Solves the singly linearly box contrained quadratic problem:

    min { (1/2)x'Qx + qx : ax = 0, 0 <= x <= C }

with
Q = [ K -K
     -K  K ]
q = [epsilon - y, epsilon + y] and
a = [1 ... 1, -1 ... -1]
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
def compute_x_r(d1, d2, lmb, u, n):
    """Compute the optimal value for x given lambda"""
    
    # x = d + lmb * a
    # with a = [1...1,-1...-1]
    x1 = d1 + lmb
    x2 = d2 - lmb
    
    # 'Apply' the box constraints
    for i in range(n):
        if(x1[i] > u): x1[i] = u
        elif(x1[i] < 0): x1[i] = 0
        
    for i in range(n):
        if(x2[i] > u): x2[i] = u
        elif(x2[i] < 0): x2[i] = 0
       
    # r = a'x
    # with a = [1...1,-1...-1]
    r = np.sum(x1) - np.sum(x2)
    
    return x1, x2, r

@jit(nopython=True)
def project(d1, d2, u, lmb, d_lmb, eps):
    """ Return the projection of the point d over the feasible region
    defined by 0 <= x <= u and ax = 0 with a = [1...1,-1...-1]
    """
    n = len(d1)
    # BRACKETING PHASE -----
    # ----------------------
    
    # Compute x and r and check whether it found the minimum
    x1, x2, r = compute_x_r(d1, d2, lmb, u, n)
    if(abs(r) < eps): return x1, x2
    
    if(r < 0):
        # r(λ) < 0 -> search for a λ | r(λ) > 0
                
        # initialize lower bounds and update
        lmb_l = lmb; r_l = r;
        lmb += d_lmb
        x1, x2, r = compute_x_r(d1, d2, lmb, u, n)
        if(abs(r) < eps): return x1, x2
        
        while(r < 0):
            # update lower bounds and lambda
            lmb_l = lmb; r_l = r;
            s = max(r_l/r -1, 0.1); d_lmb += d_lmb/s; lmb += d_lmb
            
            # Compute x and r and check whether it found the minimum
            x1, x2, r = compute_x_r(d1, d2, lmb, u, n)
            if(abs(r) < eps): return x1, x2
        
        # initialize upper bounds
        lmb_u = lmb; r_u = r

    else:
        # r(λ) > 0 -> search for a λ' | r(λ') < 0
        
        # initialize upper bounds and update lambda
        lmb_u = lmb; r_u = r; lmb -= d_lmb
        
        # Compute x and r and check whether it found the minimum
        x1, x2, r = compute_x_r(d1, d2, lmb, u, n)
        if(abs(r) < eps): return x1, x2
        
        while(r > 0):
            # update upper bounds and lambda
            lmb_u = lmb; r_u = r
            s = max(r_u/r -1, 0.1); d_lmb += d_lmb/s; lmb -= d_lmb
            
            # Compute x and r and check whether it found the minimum
            x1, x2, r = compute_x_r(d1, d2, lmb, u, n)
            if(abs(r) < eps): return x1, x2
        
        # initialize lower bounds
        lmb_l = lmb; r_l = r
    
    #secant phase
    s = 1 - r_l/r_u
    d_lmb = d_lmb/s
    lmb = lmb_u - d_lmb
    x1, x2, r = compute_x_r(d1, d2, lmb, u, n)

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

        x1, x2, r = compute_x_r(d1, d2, lmb, u, n)
                
    return x1, x2

#@njit
def project1(d1, d2, x1, x2, u, n):
    active_indeces1 = [(x1[i] == 0 and d1[i] < 0) or (x1[i] == u and d1[i] > 0) for i in range(n)]
    active_indeces2 = [(x2[i] == 0 and d2[i] < 0) or (x2[i] == u and d2[i] > 0) for i in range(n)]
    
    temp1 = None
    temp2 = None
    
    test = 0
    while(test != -1):
        d_sum = 0
        temp1 = np.array(d1)
        for i in range(n):
            if(active_indeces1[i]):
                temp1[i] = 0
            else:
                d_sum += d1[i]

        temp2 = np.array(d2)
        for i in range(n):
            if(active_indeces2[i]):
                temp2[i] = 0
            else:
                d_sum -= d2[i]

        den = (2*n - sum(active_indeces1) - sum(active_indeces2))
        if(den == 0):
            return np.zeros(n), np.zeros(n)
        else:
            v = d_sum/den
        #print(f"{den} ", end='')
            
        #v = d_sum/(2*n - sum(active_indeces1) - sum(active_indeces2))
        for i in range(n):
            if not active_indeces1[i]:
                temp1[i] = d1[i] - v
        for i in range(n):
            if not active_indeces2[i]:
                temp2[i] = d2[i] + v
    
        test = -1
        for i in range(n):
            if (x1[i] == 0 and temp1[i] < 0) or (x1[i] == u and temp1[i] > 0):
                active_indeces1[i] = True
                test = i
                break
    
        if (test != -1):
            continue

        for i in range(n):
            if (x2[i] == 0 and temp2[i] < 0) or (x2[i] == u and temp2[i] > 0):
                active_indeces2[i] = True
                test = i
                break
    
    return temp1,temp2

@njit
def equal(x, b, tol):
    return np.abs(x-b) < tol
@njit
def active_set(x, u, tol=1e-6):
    """
    Retrun arrays containing indices in the active set.
    """
    Zero = [ i for (i,e) in enumerate(equal(x,0,tol)) if e ]
    U = [ j for (j,e) in enumerate(equal(x,u, tol)) if e ]
    return set(Zero), set(U)


def active_set_changes(Z_old, U_old, Z, U):
    U_in = U - U_old
    U_out = U_old - U
    
    Z_in = Z - Z_old
    Z_out = Z_old - Z
    
    return (len(Z_in) > 0 or len(Z_out) > 0), (len(U_in) > 0 or len(U_out) > 0)



def SLBQP(K, y, C, epsilon, eps=1e-6, maxIter=1000, lmb0=0, d_lmb=2, prj_eps=1e-6, verbose=False, prj_type=1):
    """
    """
    
    n = len(y)
    
    # Input check - - - - - - - - - - -
    (d1,d2) = K.shape
    if d1 != n or d2 != n:
        raise ValueError("Q has wrong size")

    if len(y) != n:
        raise ValueError("q has wrong size")

    if not isinstance(C, np.float64):
        C = np.float64(C)
    # End of input check - - - - - - - - -
    
    # Initialization - - - - - - - - - - -
    Z_old = set()
    U_old = set()
    i = 1
    
    # x = [x1, x2]
    #x1 = np.zeros(n)
    #x2 = np.zeros(n)
    x1 = np.full(n, C/2)
    x2 = np.full(n, C/2)
    
    # q = [q1, q2]
    q1 = epsilon - y
    q2 = epsilon + y
    # - - - - - - - - - - - - - - - - - -
    
    if verbose:
        print("Iter.\tFunction val\t||gradient||\t||direction||\t  |Z|\t  |U|\tStepsize\tMaxStep")
    
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
            d1, d2 = project(x1 - g1, x2 - g2, C, lmb0, d_lmb, prj_eps)
            d1 = d1 - x1
            d2 = d2 - x2
        else:
            d1, d2 = project1(-g1, -g2, x1, x2, C, n)
            #dg = d1 @ g1 + d2 @ g2
            #print(f" ({dg}) ", end='')
        
        # Compute the norm of the gradient (g) and of the direction (d)
        g_norm = np.sqrt((g1 @ g1) + (g2 @ g2))
        d_norm = np.sqrt((d1 @ d1) + (d2 @ d2))
        
        # Print stats - - - - - - - - - - -
        if verbose:
            print("%5d\t%1.8e\t%1.8e\t%1.8e" % (i, v, g_norm, d_norm), end="")
            
#            Z, U = active_set(x, C, prj_eps)
#            Z_changed, U_changed = active_set_changes(Z_old, U_old, Z, U)
#
#            if Z_changed:
#                print("\t%5d*" % (len(Z)), end="")
#            else:
#                print("\t%5d" % (len(Z)), end="")
#
#            if U_changed:            
#                print("\t%5d*" % (len(U)), end="")
#            else:
#                print("\t%5d" % (len(U)), end="")
        # - - - - - - - - - - - - - - - - -
        
        # Check for termination
        if(d_norm < eps):
            if verbose :
                print("")
            return ('optimal', np.block([x1, x2]), v)
        if(maxIter != -1 and i >= maxIter):
            if verbose:
                print("")
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
#        if verbose:
#            Z_old = Z
#            U_old = U