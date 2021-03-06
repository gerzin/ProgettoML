from myutils import *
import numpy as np
from SLBQP import SLBQP
from numba import jit
import pickle

@jit(nopython=True)
def compute_rbf_matrix(dataset, gamma=1):
    n = len(dataset)
    K = np.empty((n,n), np.float64)
    for i in range(n):
        for j in range(i,n):
            #inlined rbf
            diff = dataset[i] - dataset[j]
            a = np.dot(diff, diff)
            v = np.exp(-gamma*a)
            K[i][j] = v
            K[j][i] = v
    return K

@jit(nopython=True) 
def rbf(x,y, gamma=1):  
    diff = x-y
    a = np.dot(diff, diff)  
    return np.exp(-gamma*a) 


class SVR:
    """
    Support Vector Regressor
    """
    def __init__(self, ker=rbf, gamma=1, C=1, eps=0.1, tol=1e-3, maxIter=1000):
        self.ker = ker
        self.gamma = gamma
        self.C = C
        self.eps = eps
        self.tol = tol
        self.maxIter = maxIter
        self.data = None
        self.gammas = None
        self.bias = 0
    
    def fit(self,X, y):
        """ Train the model with the input patterns in X and the output targets in y
        """
        K = compute_rbf_matrix(X, self.gamma)
        Q, q, a = self._prepare(K, y)

        x = np.full(len(q), self.C/2)
        status, x, e = SLBQP(Q, q, self.C, a, x, self.tol, self.maxIter)
        #status, x, e = 'a', SLBQP(Q, q, self.C, a, x, self.tol, self.maxIter), 0
        if status == 'terminated':
            print("maxIter reached")

        self.data = X
        self.gammas = self._compute_gammas(x)
        self.bias = self._compute_bias(y)

    def predict(self, pattern):
        """ Return the estimated output value for the given pattern
        """
        d = self.bias
        for i in range(len(self.data)):
            d+= self.gammas[i]*self.ker(pattern,self.data[i], self.gamma)
        return d

    def _prepare(self, K, d):
        """ Used internally to prepare the quadratic problem data
        """
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
        """ Compute the value of the Lagrangian multipliers using the solution of
        the quadratic problem
        """
        n = int(len(x)/2)
        a = x[:n]
        a1 = x[n:]
        return a-a1
    
    def _compute_bias(self, y):
        """ Compute the optimal bias
        """
        cont = 0
        bias = 0
        for i in range(len(self.gammas)):
            gamma = self.gammas[i]
            if 0<= gamma < self.C:
                cont += 1
                bias += y[i] - self.predict(self.data[i])
        if(cont == 0):
            print("Warning: No support vectors")
            return y[0] - self.predict(self.data[0])
        else:
            return bias/cont

    
    def __repr__(self):
        s = "SVR:\n"
        paramNames = ["kernel", "gamma", "C", "eps", "tol", "maxIter", "data", "gammas", "bias"]
        infos = [str(x) for x in \
                    [self.ker.__name__, self.gamma, self.C, self.eps, self.tol, self.maxIter, self.data, self.gammas, self.bias]]
        
        return s + "\n".join([f"{a}={b}" for a,b in zip(paramNames, infos)])

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
