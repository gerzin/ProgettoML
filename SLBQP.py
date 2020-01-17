import numpy as np
import pdb
from myutils import *

def compute_x(d, lmb, a, u):
    x = d + lmb*a
    for i in range(len(x)):
        if(x[i] > u): x[i] = u
        elif(x[i] < 0): x[i] = 0
    return x

def project(d, u, a, lmb, d_lmb, eps=1e-6):
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
    
def SLBQP(Q, q, u, eps=1e-6, maxIter=1000): 
    n = int(len(q)/2)
    x = np.full(2*n, u/2)
    i = 1
    
    a = np.empty(2*n)
    a[0:n] = np.ones(n)
    a[n:] = - np.ones(n)
    
    while True:
        # Compute function value (v), gradient (g) and descent direction (d)
        Qx = np.dot(Q, x)

        #pdb.set_trace()
        v = np.dot(Qx,x) + np.dot(q, x)
        g = np.array(Qx + q)
        d = np.array(x-g).flatten()
    
        # Project the direction over the feasible region
        d = project(d, u, a, 0, 2)
        d = d - x
        
        d_norm = np.linalg.norm(d)
        print(f"anti-gradient:\n{-g}")
        print(f"direction:\n{d}")
        print(f"norm:\t{d_norm}")
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
            alpha = min(max_alpha, np.dot(d, d)/den)
            
        #print(f"max_alpha:\t{max_alpha}")
        #print(f"den:\t{den}")
        #print(f"-g*d:\t{np.dot(-g,d)}")
        #print(f"alpha:\t{alpha}")
        
        x = x + alpha * d
        
        i = i + 1
