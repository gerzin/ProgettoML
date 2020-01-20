import numpy as np
import pdb
from myutils import *
from numba import jit

@jit(nopython=True, parallel=True)
def compute_x(d, lmb, a, u):
    x = d + lmb*a
    for i in range(len(x)):
        if(x[i] > u): x[i] = u
        elif(x[i] < 0): x[i] = 0
    return x


@jit(nopython=True)
def project(d, u, a, lmb, d_lmb, eps=1e-6):
    """
    
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

#@jit(nopython=False)
def SLBQP(Q, q, u, a, x, eps=1e-6, maxIter=1000): 
    """

    """
    assert len(q)%2 == 0
    n = int(len(q)/2)
    i = 1
    
    
    while True:
        # Compute function value (v), gradient (g) and descent direction (d)
        Qx = np.dot(Q, x)

        #pdb.set_trace()
        v = np.dot(Qx,x) + np.dot(q, x)
        g = np.array(Qx+q)
        
        d = np.array(x-g)
    
        # Project the direction over the feasible region
        d = project(d, u, a, 0, 2)
        d = d - x
        
        d_norm = np.linalg.norm(d)
        if(d_norm < eps):
            return ('optimal', x, v)
        if(i >= maxIter):
            return ('terminated', x, v)
        
        max_alpha = np.Inf
        for j in range(n):
            if(d[j] > 0):
                max_alpha = min( max_alpha, (u - x[j])/d[j] )
            elif(d[j] < 0):
                max_alpha = min( max_alpha, (-x[j])/d[j] )
        
        # den = d' * Q * d
        den = np.dot(d, np.dot(Q, d))
        if(den <= 1e-16):
            alpha = max_alpha
        else:
            alpha = min(max_alpha, (d_norm**2)/den)
            
                
        x = x + alpha * d
        
        i = i + 1