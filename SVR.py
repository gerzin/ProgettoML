from myutils import *
import numpy as np
from SLBQP import SLBQP

def linear(x,y):
    return np.dot(x,y)

def rbf(x,y, gamma=1):
    a = np.dot(x-y, x-y)
    return np.exp(-gamma*a)



class SVR:
    """
    Support Vector Regressor
    """
    
    def __init__(self, ker=rbf, gamma=1, C=1, eps=0.1, tol=1e-3, maxIter=1000):
        self.ker = rbf
        self.gamma = gamma
        self.C = C
        self.eps = eps
        self.tol = tol
        self.maxIter = maxIter
    
    def fit(self,X, y):
        K = self._compute_kernel_matrix(X, self.ker)
        Q, q, a = self._prepare(K, y)
        status, x, e = SLBQP(Q, q, self.C)
        if status == 'terminated':
            print("maxIter reached")
        self.data = X
        self.gammas = self._compute_gammas(x)

    def predict(self, pattern):
        d = 0
        for i in range(len(pattern)):
            d+= self.gammas[i]*self.ker(pattern,self.data[i])
        return d

    def _compute_kernel_matrix(self,dataset, dot_product=linear):
        n = len(dataset)
        K = np.empty([n,n])
        
        for i in range(n):
            for j in range(i,n):
                v = dot_product(dataset[i], dataset[j])
                
                K[i][j] = v
                K[j][i] = v

        return K

    def _prepare(self, K, d):
        (n,m) = K.shape
        if(n != m):
            print("matrix must be square")
            return
        
        # compute quadratic part of the Quadratic Problem
        Q = np.block([
            [K, -K],
            [-K, K]
        ])

        # compute linear part of the Quadratic Problem
        q = np.empty(2*n)
        q[:n] = self.eps - d
        q[n:] = self.eps + d
        
        # compute vector for the linear constraint ax = 0
        a = np.empty(2*n)
        a[:n] = 1.
        a[n:] = -1.
        
        return Q,q,a
    
    def _compute_gammas(self,x):
        n = int(len(x)/2)
        a = x[:n]
        a1 = x[n:]
        return a-a1
        






