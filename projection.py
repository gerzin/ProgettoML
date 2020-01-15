import numpy as np

# USED TO COMPUTE THE PROJECTION ON THE FEASIBLE SET
#
# The problem can be described as
#   min{ (1/2)z'z - d'z : 0 <= z <= C , c'z = 0 }
# where z is the projection and d is the antigradient
#
# Just a simpler version of the problem
#       min{ (1/2)x'Ax - c'x : l <= x <= u , a'x = b }
# where A = diag(d1, d2, . . ., dn) diagonal matrix
# in our case A = I identity matrix
#
# Lagrangian penalty function: φ(x,λ) = (1/2)x'Ax - c'x - λ(a'x - b)
# over the constraints: l <= x <= u
#
#




# Compute the r(λ) for the given λ
#
# φ(x,λ) = (1/2)xAx - cx - λ(ax - b)
# r(λ) = ax(λ) - b
# x(λ) = mid(l, h, u) with hi = (ci + λai)/di
#
def compute_r(c,lmb,a,b,l,u):
    x = c + lmb*a
    for i in range(len(x)):
        if(x[i] > u): x[i] = u
        elif(x[i] < l): x[i] = l
    
    return a*x + b
    

def bracketing(a,b,c,l,u, lmb0,d_lmb):
    lmb = lmb0
    r = compute_r(c, lmb, a, b, l, u)
    if(abs(r) < eps) return
    
    if(r < 0):
        lmb_l = lmb
        r_l = r
        lmb = lmb + d_lmb
        r = compute_r(c, lmb, a, b, l, u)
        if(abs(r) < eps) return
        
        while(r < 0):
            lmb_l = lmb
            r_l = r
            # ...
        lmb_u = lmb
        r_u = r
    else:
    