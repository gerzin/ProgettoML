import pathlib
import argparse
import numpy as np
from time import time
from numba import jit, njit, prange
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers

import time

def solve_with_cvxopt(K, target, epsilon, u):
    """Builds and solves the problem with the cvxopt primal-dual solver
    Params:
        K
        target
        epsilon
        u
    Returns:
        sol     --  object containing the primal and dual solutions.
    """
    
    size = len(target)

    Q = matrix(np.block([[K, -K], [-K, K]]))
    
    q = matrix(np.block([epsilon - target, epsilon + target]))
    
    G = matrix(np.block([[np.eye(2*size)], [-np.eye(2*size)]]))
    
    h = np.zeros(4*size)
    h[0:2*size] = u
    h = matrix(h)

    A = matrix(np.block([[np.ones(size), -np.ones(size)]]))
    
    b = matrix(np.zeros(1))
    sol = solvers.qp(Q, q, G, h, A, b)
    return sol['dual objective']
    # return sol




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


# Togliere ?
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
    #np.random.seed(seed)
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
    return data.data, data.target

def plot_multiple_functions(functions, plot_avg=False, ax=None, color=None, col_avg=None, label="average"):
    plt = None
    if ax is not None:
        plt=ax
    else:
        plt = matplotlib.pyplot
    for points in functions:
        col = "cornflowerblue" if color is None else color
        plt.plot([*range(1,1+len(points))], points, color=col)
    if plot_avg:
        """
        calcola la lunghezza "m" della lista più piccola e poi calcola la media
        considerando i primi m elementi di ogni lista
        """
        minlength = min(map(lambda x : len(x), functions))
        cropped_functions = map(lambda x : x[:minlength], functions)
        nfun = len(functions)
        average = [sum(i)/nfun for i in zip(*cropped_functions)]
        col_avg = "blue" if col_avg is None else col_avg
        plt.plot([*range(1, 1+minlength)], average, label=label, color=col_avg)
        plt.legend()
    if ax is None:
        plt.show()
