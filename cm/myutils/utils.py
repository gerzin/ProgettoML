import pathlib
import argparse
import numpy as np
from numba import jit, njit, prange
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers


def solve_with_cvxopt(K, target, epsilon, u):
    """Builds and solves the problem with the cvxopt primal-dual solver
    Params:
        K       -- kernel matrix used as block to build Q
        target  -- target vector to build vector q
        epsilon -- coefficient used to build vector q
        u       -- upper bound for the feasible region
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
    return sol



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
    DATASET_PATH = __file__[:-16] + "data"
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