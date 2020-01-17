import argparse
import numpy as np
from time import time

def print_invocation(f):
    def wrapper(*args, **kwargs):
        print(f"{f.__name__} called")
        ret = f(*args, **kwargs)
        print(f"{f.__name__} returned {ret}")
        return ret
    return wrapper

def dump_args(f):
	argnames = f.__code__.co_varnames
	def wrapper(*args, **kwargs):
		argval = ','.join('%s=%r' % entry for entry in zip(argnames, args) )
		print(f"{f.__name__}({argval})")
		return f(*args, **kwargs)
	return wrapper


def time_it(f):
    def wrapper(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"{f.__name__} took {te-ts}")
        return result
    return wrapper

def get_cmdline_args():
    parser = argparse.ArgumentParser(description='Support Vector Regression using Gradient Projection.')    
    parser.add_argument('-f','--file', help='input csv file')
    return parser.parse_args()

def load_data(csvfile, delfirst=True):
    loaded_data = np.loadtxt(csvfile, dtype=np.float64, delimiter=',')
    if delfirst:
        loaded_data = np.delete(loaded_data, 0, 1)
    Y_1, Y_2 = loaded_data[:,-2], loaded_data[:,-1]
    loaded_data = np.delete(np.delete(loaded_data, -1, 1), -1, 1)
    return loaded_data, Y_1, Y_2


def build_problem(n, u):
    n2 = int(n/2)
    G = np.block([[np.eye(n)], [-np.eye(n)]])
    A = np.block([ [np.ones(n2), -np.ones(n2) ]])
    h = np.zeros(2*n)
    h[0:n] = u
    b = np.zeros(1)
    return G, A, h, b

def linear(x,y):
    return np.dot(x,y)

def compute_kernel_matrix(dataset, dot_product=linear):
    n = len(dataset)
    K = np.empty([n,n])
    
    for i in range(n):
        for j in range(i,n):
            v = dot_product(dataset[i], dataset[j])
            
            K[i][j] = v
            K[j][i] = v

    return K

def prepare(K, eps, d, C):
    (n,m) = K.shape
    if(n != m):
        print("matrix must be square")
        return
    
    # compute quadratic part of the Quadratic Problem
    Q = np.empty([2*n,2*n])
    Q[:n][:n] = K
    Q[:n][n:] = -K
    Q[n:][:n] = -K
    Q[n:][n:] = K
    
    # compute linear part of the Quadratic Problem
    q = np.empty(2*n)
    q[:n] = eps - d
    q[n:] = eps + d
    
    # compute vector for the linear constraint ax = 0
    a = np.empty(2*n)
    a[:n] = 1.
    a[n:] = -1.
    
    return Q,q,C,a