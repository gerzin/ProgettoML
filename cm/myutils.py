import pathlib
import argparse
import numpy as np
from time import time
from numba import jit, njit, prange
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt


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
        argval = ','.join('%s=%r' % entry for entry in zip(argnames, args))
        print(f"{f.__name__}({argval})")
        return f(*args, **kwargs)
    return wrapper


def time_it(f):
    """Decorator that prints the time in seconds the function took to run."""
    def wrapper(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"{f.__name__} took {te-ts}")
        return result
    return wrapper


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
        Y_1, Y_2 = loaded_data[:, -2], loaded_data[:, -1]
        loaded_data = np.delete(np.delete(loaded_data, -1, 1), -1, 1)
        return loaded_data, Y_1, Y_2
    else:
        return loaded_data


# @jit(nopython=True)
def shuffleRows(M):
    """Shuffles the rows of a matrix"""
    np.random.shuffle(M)


@jit(nopython=True)
def splitHorizontally(matrix, percentage):
    """Split matrix horizontally and returns the two matrices."""
    assert 0 <= percentage <= 1
    lenM1 = round(matrix.shape[0]*percentage)
    return matrix[0:lenM1], matrix[lenM1:]


def separate_feature(df, nlast=1):
    """separates the features from the target variables.
    Params:
        dataset  -- the dataset.
        nlast    -- number of columns containing the target variables.
    Returns:
        features, targets
    """
    target_points = df[df.columns[-nlast:]]
    #target_points.columns = ['X', 'Y']
    feature_points = df[df.columns[:-nlast]]
    feature_points.columns = [str(i+1)
                              for i in range(len(feature_points.columns))]
    return feature_points, target_points


def randomsample(mat, n):
    """samples a random subset of size n from a matrix."""
    r, c = mat.shape
    assert(r >= n)
    M = mat.to_numpy() if type(mat) is pd.core.frame.DataFrame else mat
    return M[np.random.choice(r, n, replace=False), :]


def sample_transform_problem(feature, target, size, seed=None):
    """
    samples a subset of the input matrices/vectors and applies a kernel.
    """
    np.random.seed(seed)
    r, c = feature.shape
    assert(r >= size)
    rand_ind = np.random.choice(r, size, replace=False)

    M = feature.to_numpy() if type(feature) is pd.core.frame.DataFrame else feature
    T = target.to_numpy() if type(target) is pd.core.frame.DataFrame else target
    if len(T.shape) > 1:
        if(T.shape[1] == 1): T = T.flatten()

    

    featuresamp = M[rand_ind, :]
    targetsamp = T[rand_ind]

    K = compute_kernel_matrix(featuresamp, rbf)
    return K, targetsamp


def build_problem(n, u):
    """
    """
    n2 = int(n/2)
    G = np.block([[np.eye(n)], [-np.eye(n)]])
    A = np.block([[np.ones(n2), -np.ones(n2)]])
    h = np.zeros(2*n)
    h[0:n] = u
    b = np.zeros(1)
    return G, A, h, b


@jit(nopython=True)
def linear(x, y):
    return np.dot(x, y)


@jit(nopython=True)
def rbf(x, y, gamma=1):
    a = np.dot(x-y, x-y)
    return np.exp(-gamma*a)


def compute_kernel_matrix(dataset, dot_product=linear):
    n = len(dataset)
    K = np.zeros([n, n])

    for i in range(n):
        for j in range(i, n):
            v = dot_product(dataset[i], dataset[j])

            K[i, j] = v
            K[j, i] = v

    return K


@jit(nopython=True)
def scale(arr):
    M, m = max(arr), min(arr)
    scaled = (arr-m)/(M - m)
    return scaled, M, m


@jit(nopython=True)
def scale_back(scaled, M, m):
    return scaled*(M-m)+m


# colors
RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"


def print_decreasing(ite, v, g_norm, d_norm, old=None):
    formatter = ""
    if old is None:
        formatter = "%5d\t{}%1.16e\t{}%1.16e\t{}%1.16e\033[0;0m".format(
            GREEN, GREEN, GREEN)
    else:
        new = (ite, v, g_norm, d_norm)
        values = []
        for i in range(4):
            if old[i] - new[i] > 0:
                values.append(GREEN)
            else:
                values.append(RED)
        formatter = "%5d\t{0}%1.16e\t{1}%1.16e\t{2}%1.16e\033[0;0m".format(
            values[1], values[2], values[3])
    print(formatter % (ite, v, g_norm, d_norm), end="")
    return (ite, v, g_norm, d_norm)


def dump_on_file(filename):
    """Decorator that appends the result of f on a file."""
    def decorator(function):
        def wrapper(*args, **kw):
            result = function(*args, **kw)
            with open(filename, "a") as f:
                writer = csv.writer(f)
                writer.writerow(result)
            return result
        return wrapper
    return decorator


def load_ml_dataset():
    DATASET_PATH = __file__[:-10] + "data"
    DATASET_NAME = "ML-CUP19-TR.csv"
    print("loading from: ")
    print(DATASET_PATH + "/" + DATASET_NAME)
    df = pd.read_csv(DATASET_PATH + "/" + DATASET_NAME, sep=',',
                     comment='#', skiprows=7, header=None, index_col=0)
    features, targets = separate_feature(df, 2)

    return features.to_numpy(), targets.to_numpy()


def load_airfoil_dataset():
    DATASET_PATH = __file__[:-10] + "data"
    DATASET_NAME = "airfoil_self_noise.csv"

    print(DATASET_PATH + "/" + DATASET_NAME)
    df = pd.read_csv(DATASET_PATH + "/" + DATASET_NAME)
    df.columns = ['Frequency (HZ)', 'Angle of attack (deg)', 'Chord length (m)', 'Free-stream velocity (m/s)',
                  'Suction side displacement thickness (m)', 'Scaled sound pressure level (db)']
    df, targets = separate_feature(df, 1)
    return df.to_numpy(), targets.to_numpy()


def load_california_dataset():
    from sklearn.datasets.california_housing import fetch_california_housing
    data = fetch_california_housing()
    return pd.DataFrame(data.data, columns=data.feature_names)

def plot_multiple_functions(functions, plot_avg=False, ax=None):
    plt = None
    if ax is not None:
        plt=ax
    else:
        plt = matplotlib.pyplot
    for points in functions:
        plt.plot([*range(len(points))], points, color="cornflowerblue")
    if plot_avg:
        """
        calcola la lunghezza "m" della lista più piccola e poi calcola la media
        considerando i primi m elementi di ogni lista
        """
        minlength = min(map(lambda x : len(x), functions))
        cropped_functions = map(lambda x : x[:minlength], functions)
        nfun = len(functions)
        average = [sum(i)/nfun for i in zip(*cropped_functions)]
        plt.plot([*range(minlength)], average, label="average", color="blue")
        plt.legend()
    if ax is None:
        plt.show()
