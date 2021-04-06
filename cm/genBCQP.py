#!/usr/bin/env python3
import numpy as np
import numpy.linalg as la

def genBCQP(n , actv=0.5 , rank=1.1 , ecc=0.99, u=10, seed=None):
    """
    Generates a box constrained quadratic program.  
    Params:
        n       --  the size of the problem
        actv    --  how many box constraints (as a fraction of n) the unconstrained optimum will  
                    violate and so we expect to be active (but there's no guarantee)
        ecc     --  eccentricity (λ_min - λ_max)/(λ_min + λ_max)
        rank    --  if > 1 Q can be expected to be full rank
        u       --  max val of each u_i
        seed    --  seed for the random number generator
    Returns:
        Q, q, a
    """
    n = int(n)

    Q, q = None, None
    
    np.random.seed(seed)

    # Generate Q
    G = np.random.rand(round(rank * n) , n)
    Q = np.transpose(G) @ G
    
    d, V = la.eig(Q)
    d = np.sort(d)
    
    l = (d - d[1])*(2*ecc)/(1-ecc)*d[1]/(d[n-1]-d[1]) + d[1]
    Q = np.dot( np.dot(V, np.diag(l)), la.inv(V) )

    # Generate q
    z = np.zeros(n)
    outb = [ i <= actv for i in np.random.rand(n)]
    
    lr = [i <= 0.5 for i in np.random.rand(n)]
    
    l = [a and b for (a,b) in zip(outb, lr)]
    r = [a and not b for (a,b) in zip(outb, lr)]
    

    for (i,j) in enumerate(l):
        if j:
            z[i] = -u*np.random.random_sample()

    for (i,j) in enumerate(r):
        if j:
            z[i] = u*(1 + np.random.random_sample())

    outb = [not i for i in outb]
    for (i,j) in enumerate(outb):
        if j:
            z[i] = np.random.random_sample()*u


    q = -Q @ z

    return Q, q

if __name__ == "__main__":
    n = 5
    Q,q,a = genBCQP(n)
    print(Q)
    print(q)