from myutils import *
import numpy as np
from SLBQP import SLBQP
from numba import jit, jitclass, int32, float32, float64
import pickle

def linear(x,y):
    return np.dot(x,y)

@jit(nopython=True)
def rbf(x,y, gamma=1):
    a = np.dot(x-y, x-y)
    return np.exp(-gamma*a)


@jit(nopython=True)
def compute_K_matrix(dataset, gamma=1):
    n = len(dataset)
    K = np.empty((n,n), np.float64)
    for i in range(n):
        for j in range(i,n):
            #inlined rbf
            diff = dataset[i] - dataset[j]
            a = np.dot(diff, diff)
            v = np.exp(-gamma*a)
            #v = rbf(dataset[i], dataset[j])
            K[i][j] = v
            K[j][i] = v
    return K

#@jit(nopython=True)
def MSE(outputs, targets):
    return np.square(outputs - targets).mean()


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
        self.data = None
        self.gammas = None
        self.bias = 0
    
    def fit(self,X, y):
        #K = self._compute_kernel_matrix(X, self.ker)
        K = compute_K_matrix(X)
        Q, q, a = self._prepare(K, y)
        x = np.full(len(q), self.C/2)
        status, x, e = SLBQP(Q, q, self.C, a, x, self.tol, self.maxIter)
        if status == 'terminated':
            print("maxIter reached")
        self.data = X
        self.gammas = self._compute_gammas(x)
        self.bias = self._compute_bias(y)

    def predict(self, pattern):
        d = self.bias
        for i in range(len(self.data)):
            d+= self.gammas[i]*self.ker(pattern,self.data[i])
        return d

    def _compute_kernel_matrix(self,dataset, dot_product):
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

    def save(self, filename):
        """Save the regressor parameters on a file"""
        with open(filename, "wb") as filename:
            pickle.dump(self,filename)
    
    @staticmethod
    def load(filename):
        """load the regressor parameters from a file"""
        with open(filename, "rb") as filename:
            r = pickle.load(filename)
        return r
    
    def _compute_bias(self, y):
        cont = 0
        bias = 0
        for i in range(len(self.gammas)):
            gamma = self.gammas[i]
            if 0< gamma < self.C:
                cont += 1
                bias += y[i] - self.predict(self.data[i])
        
        return bias/cont
    
    def __repr__(self):
        s = "SVR:\n"
        paramNames = ["kernel", "gamma", "C", "eps", "tol", "maxIter", "data", "gammas", "bias"]
        infos = [str(x) for x in \
                    [self.ker.__name__,self.gamma, self.C, self.eps, self.tol, self.maxIter, self.data, self.gammas, self.bias]]
        
        return s + "\n".join([f"{a}={b}" for a,b in zip(paramNames, infos)])

    def evaluateMSE(self, patterns, targets):
        outputs = [self.predict(x) for x in patterns]
        return MSE(outputs, targets)
        
      





