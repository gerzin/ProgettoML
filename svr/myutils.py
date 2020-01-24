import argparse
import numpy as np
from time import time
from numba import jit
import csv

def print_invocation(f):
    """Decorator that prints when a function has been invocated and when it returns.
    
    It also prints the return value.
    """
    def wrapper(*args, **kwargs):
        print(f"{f.__name__} called")
        ret = f(*args, **kwargs)
        print(f"{f.__name__} returned {ret}")
        return ret
    return wrapper

def dump_args(f):
    """Decorator that prints when a function has been invocated and its parameters."""
	argnames = f.__code__.co_varnames
	def wrapper(*args, **kwargs):
		argval = ','.join('%s=%r' % entry for entry in zip(argnames, args) )
		print(f"{f.__name__}({argval})")
		return f(*args, **kwargs)
	return wrapper

def dump_svr_params(filename, tuples):
    """Write (append) on a file a tuple in csv format.
    
    Params:
        filename    -- name of the file.
        tuples      -- the tuple to write
    """
    with open(filename, "a") as csvfile:
        csv_out = csv.writer(csvfile)
        csv_out.writerow(tuples)

def time_it(f):
    """Decorator that prints the time in seconds the function took to run."""
    def wrapper(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"{f.__name__} took {te-ts}")
        return result
    return wrapper

def get_cmdline_args(descr='Support Vector Regression using Gradient Projection.'):
    """Utility to parse the command line arguments.

    It returns an object containing a mapping <param name, param value>.
    """
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('-f','--file', help='input csv file', required=True)
    parser.add_argument('-p', '--percentage', help="percentage", type=float)
    parser.add_argument('-c', '--column', help="target column (1 or 2)", type=int, choices=[1,2], required=True)
    parser.add_argument('-k', '--kfold', help="parameter for k-fold validation", type=int)
    parser.add_argument('-s', '--scale', help="scale the data", action='store_true')
#    parser.add_argument('-a', '--arguments',  nargs='+', help='gamma C eps tol', required=False)
    return parser.parse_args()

#@jit(nopython=True)
def load_data(csvfile, delfirst=True, shuffle=False, split=True):
    """Load a matrix from a csv file.
    
    Params:
        delfirst    -- delete the first column of the matrix.
        shuffle     -- shuffle the rows of the matrix
        split       -- separate the last two column from the main matrix
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
    """

    """
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
    """Prepare thr problem."""
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

def extract_best_configs(filename, n=1):
    """Extract the best n configurations from a file.
    Params:
        filename    -- name of the csv file.
        n           -- number of config to extract
    """
    f = np.loadtxt(filename, delimiter=',')
    f = list(filter(lambda x : x[4] >= 0, f))
    s = sorted(f, key = lambda x : x[4]) #sort error-wise
    return s[0:n]
