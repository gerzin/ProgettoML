import argparse
import numpy as np
from time import time
from numba import jit

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
    parser.add_argument('-p', '--percentage', help="percentage ")
    return parser.parse_args()

#@jit(nopython=True)
def load_data(csvfile, delfirst=True, shuffle=False, split=True):
    """Load a matrix from a csv file.
    
    Params:
        delfirst -- delete the first column of the matrix.
        shuffle -- shuffle the rows of the matrix

    """
    loaded_data = np.loadtxt(csvfile, dtype=np.float64, delimiter=',')
    if delfirst:
        loaded_data = np.delete(loaded_data, 0, 1)
    if shuffle:
        np.random.shuffle(loaded_data)
    if split:
        Y_1, Y_2 = loaded_data[:,-2], loaded_data[:,-1]
        loaded_data = np.delete(np.delete(loaded_data, -1, 1), -1, 1)
        return loaded_data, Y_1, Y_2
    else:
        return loaded_data

@jit(nopython=True)
def shuffleRows(M):
    """Shuffles the rows of a matrix"""
    np.random.shuffle(M)

@jit(nopython=True)
def splitHorizontally(matrix, percentage):
    """Split matrix horizontally and returns the two matrices."""
    assert 0<= percentage <= 1
    lenM1 = round(matrix.shape[0]*percentage)
    return matrix[0:lenM1], matrix[lenM1:]

@jit(nopython=True)
def build_problem(n, u):
    n2 = int(n/2)
    G = np.block([[np.eye(n)], [-np.eye(n)]])
    A = np.block([ [np.ones(n2), -np.ones(n2) ]])
    h = np.zeros(2*n)
    h[0:n] = u
    b = np.zeros(1)
    return G, A, h, b

@jit(nopython=True)
def linear(x,y):
    return np.dot(x,y)

@jit(nopython=True)
def rbf(x,y, gamma=1):
    a = np.dot(x-y, x-y)
    return np.exp(-gamma*a)

@jit(nopython=True)
def compute_kernel_matrix(dataset, dot_product=linear):
    n = len(dataset)
    K = np.empty([n,n])
    
    for i in range(n):
        for j in range(i,n):
            v = dot_product(dataset[i], dataset[j])
            
            K[i][j] = v
            K[j][i] = v

    return K

@jit(nopython=True)
def scale(arr):
    M, m = max(arr), min(arr)
    scaled = (arr-m)/(M - m)
    return scaled, M, m

@jit(nopython=True)
def scale_back(scaled, M, m):
    return scaled*(M-m)+m

@jit(nopython=True)
def prepare(K, eps, d, C):
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
    q[:n] = eps - d
    q[n:] = eps + d
    
    # compute vector for the linear constraint ax = 0
    a = np.empty(2*n)
    a[:n] = 1.
    a[n:] = -1.
    
    return Q,q,C,a

