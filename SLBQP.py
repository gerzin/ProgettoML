import numpy as np
import pdb
from myutils import *

def compute_x(d, lmb, a, u):
    x = d + lmb*a
    for i in range(len(x)):
        if(x[i] > u): x[i] = u
        elif(x[i] < 0): x[i] = 0
    return x

@print_invocation
def project(d, u, a, lmb, d_lmb, eps=1e-6):
    # BRACKETING PHASE -----
    # ----------------------
    
    # Compute x and r and check whether it found the minimum
    x = compute_x(d, lmb, a, u); r = np.dot(a, x)
    if(abs(r) < eps): return x
    
    if(r < 0):
        # start looking for a positive value of r
        
        # update lower bounds and lambda
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
            x = compute_x(d, lmb, a, u)
            r = np.dot(a, x)
            if(abs(r) < eps): return x
        
        # initialize upper bounds
        lmb_u = lmb; r_u = r

    else:
        lmb_u = lmb; r_u = r; lmb -= d_lmb
        x = compute_x(d, lmb, a, u)
        r = np.dot(a, x)
        if(abs(r) < eps): return x
        while(r > 0):
            lmb_u = lmb; r_u = r
            s = max(r_u/r -1, 0.1); d_lmb += d_lmb/s
            lmb -= d_lmb
            x = compute_x(d, lmb, a, u)
            r = np.dot(a, x)
            print(r)
            if(abs(r) < eps): return x
        lmb_l = lmb
        r_l = r
        print(r)
    
    #secant phase
    s = 1 - r_l/r_u
    d_lmb = d_lmb/s
    lmb = lmb_u - d_lmb
    x = compute_x(d, lmb, a, u)
    r = np.dot(a, x)
    while(abs(r) >= eps):
        print(r)
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
    
    while True:
        # Compute function value (v), gradient (g) and descent direction (d)
        Qx = np.dot(Q, x)

        #pdb.set_trace()
        v = np.dot(x, Qx) + np.dot(q, x)
        g = np.array(Qx + q)
        d = x-g

        # Project the direction over the feasible region
        a = np.empty(2*n)
        a[0:n] = np.ones(n)
        a[n:] = - np.ones(n)
        #print(f"prima\n{d}")
        d = project(d, u, a, 0, 2)
        d = d - x
        #print(f"dopo\n{d}")
        d_norm = np.linalg.norm(d)
        print(f"norma\n{d_norm}")
        
        if(d_norm < eps):
            return x
        if(i >= maxIter):
            return x
        
        max_alpha = np.Inf
        for i in range(n):
            if(d[i] > 0):
                max_alpha = min( max_alpha, (u - x[i])/d[i] )
            elif(d[i] < 0):
                max_alpha = min( max_alpha, (-x[i])/d[i] )
        print(f"max_alpha\n{max_alpha}")
        
        # den = d' * Q * d
        den = np.dot(d, np.dot(Q, d))
        if(den <= 1e-16):
            alpha = max_alpha
        else:
            alpha = min(max_alpha, np.dot(-g, d)/den)
        
        x = x + alpha * d
        
        i = i + 1

        
            
Q = np.matrix("1,2,3,4;2,1,2,3;3,2,1,2;4,3,2,1")
Q = np.array([[1,2,3,4],[2,1,2,3],[3,2,1,2],[4,3,2,1]])
q = np.array([1,2,3,4])
u = 10
SLBQP(Q,q,u)
