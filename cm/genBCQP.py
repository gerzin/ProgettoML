#!/usr/bin/env python3
import numpy as np
import numpy.linalg as la

def genBCQP(n , actv=0.5 , rank=1.1 , ecc=0.99, u=10, seed=None):
    Q, q, a = None, None, None
    
    np.random.seed(seed)

    G = np.random.rand(round(rank * n) , n )
    Q = np.transpose(G) @ G
    
    d, V = la.eig(Q)
    d = np.sort(d)
    
    l = (d - d[1])*(2*ecc)/(1-ecc)*d[1]/(d[n-1]-d[1]) + d[1]
    Q = np.dot( np.dot(V, np.diag(l)), la.inv(V) )

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

    x = np.random.uniform(0,1,n)
    a = np.empty(n)
    a[1:n] = np.random.uniform(-1,1,n-1)
    a[0] = -(x[1:n] @ a[1:n])/x[0]
    a = a/la.norm(a)

    return Q, q, a

if __name__ == "__main__":
    n = 5
    Q,q,a = genBCQP(n)
    print(Q)
    print(q)
    print(a)
    

