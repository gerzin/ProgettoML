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
            if(abs(r) < eps): return x
        lmb_l = lmb
        r_l = r
    
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
    
    while True:
        # Compute function value (v), gradient (g) and descent direction (d)
        Qx = np.dot(Q, x)

        #pdb.set_trace()
        v = np.dot(Qx,x) + np.dot(q, x)
        g = np.array(Qx + q)
        d = np.array(x-g).flatten()

        # Project the direction over the feasible region
        a = np.empty(2*n)
        a[0:n] = np.ones(n)
        a[n:] = - np.ones(n)
        d = project(d, u, a, 0, 2)
        d = d - x
        d_norm = np.linalg.norm(d)
        
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
        
        # den = d' * Q * d
        den = np.dot(d, np.dot(Q, d))
        if(den <= 1e-16):
            alpha = max_alpha
        else:
            alpha = min(max_alpha, np.dot(-g, d)/den)
        
        x = x + alpha * d
        
        i = i + 1

        
            
#Q = np.matrix("1,2,3,4;2,1,2,3;3,2,1,2;4,3,2,1")
#Q = np.array([[1,2,3,4],[2,1,2,3],[3,2,1,2],[4,3,2,1]])
#q = np.array([1,2,3,4])
u = 10

Q = np.array([[6.2521  ,  4.5275  ,  5.3306  ,  2.5052  ,  3.1519  ,  3.9862],[4.5275  ,  5.2587  ,  4.2953   , 2.0428  ,  3.4669   , 3.7038],[5.3306 ,   4.2953  ,  5.4904 ,   2.0756  ,  3.0360  ,  3.4988],[2.5052  ,  2.0428  ,  2.0756  ,  1.7720  ,  1.5066  ,  1.6718],[3.1519  ,  3.4669  ,  3.0360  ,  1.5066  ,  3.5791  ,  3.1391],[3.9862  ,  3.7038  ,  3.4988  ,  1.6718  ,  3.1391  ,  3.4063]])

q = np.array([-162.0451, -147.2674,-154.0566,-76.1109,-110.5499,-117.7583])
print(SLBQP(Q,q,u))
