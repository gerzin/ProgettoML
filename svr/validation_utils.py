import numpy as np
from myutils import dump_svr_params
from SVR import SVR
from numba import jit, njit

def compute_error(svr, patterns, targets, M=None, m=None, err='mse'):
    """Estimate the error of the model.

    Params:
        svr         -- regressor class
        patterns    -- input patterns
        targets     -- output targets
        M, m        -- parameter for scaling back (see "scale" below)
        err         -- error function used to evaluare the error.
    """
    outputs = [ svr.predict(x) for x in patterns ]
    if M and m:
        outputs = [ y*(M-m) + m for y in outputs ]
    
    if err == 'mse':
        e = np.square(outputs - targets).mean()
    elif err == 'mee':
        e = np.absolute(outputs - targets).mean()
    
    return e

def compute_mee(svr1, svr2, X, y1, y2):
    """Compute the Mean Euclidean Error.
    Params:
        svr1    -- SVR for the first target column.
        svr2    -- SVR for the second target column.
        X       -- input patterns.
        y1      -- first target column.
        y2      -- second target column.
    Retval:
        The Mean Euclidean Error.
    """
    out1 = np.array([ svr1.predict(x) for x in X ])
    out2 = np.array([ svr2.predict(x) for x in X ])
    diff1 = np.square(out1-y1)
    diff2 = np.quare(out2-y2)
    return np.sqrt(diff1-diff2).mean()
    
def scale(y):
    """Scale the value of the array betweet 0 and 1.

    Retval:
        scaled  -- array whose values are scaled.
        M, m    -- parameters used to scale back the values.
    """
    M, m = max(y), min(y)
    scaled = (y - m)/(M - m)
    return scaled, M, m


def scale_back(scaled, M, m):
    """Scale back the values in the array scaled.

    Params:
        scaled  -- array containing scaled values between 0 and 1.
        M, m    -- paramenters used to scale back to the original values.
    Retval:
        an array containing the original values.
    """
    return scaled*(M - m) + m


@njit()
def k_fold_indeces(n, k):
    """Utility function returning indices used for the partition of a matrix in k sets.
    
    Param:
        n   -- dimension of the matrix.
        k   -- number of folds
    Retval:
        array containing indices.
    """
    fold_size = int(n/k)
    return [ i*fold_size for i in range(k) ] + [n]


def split_dataset(X,Y, start,end):
    """Split the dataset into training and validation set according to the given interval.

    Params:
        X           -- input patterns.
        Y           -- target values.
        start, end  -- indices where to split.
    Retval:
        tr          -- training set
        tr_y        -- target values of the training set.
        vs          -- validation set.
        vs_y        -- target values of the validation set.
    """
    vs_interval = [range(start,end)]
    
    tr = np.delete(X, vs_interval, 0)
    tr_y = np.delete(Y, vs_interval, 0)
    
    vs = X[start:end]
    vs_y = Y[start:end]
    
    return tr,tr_y, vs,vs_y


def k_fold_evaluate(X, y, k, folds, params, threshold=np.Inf, scaleY=False):
    """
    Evaluate the error of a model using k-fold cross validation. 

    It prints the parameters and the corresponding error error on a csv file.
    
    Note:
        it uses early stopping when the error is too high, in this case the error field will containing the value -1.
    Params:
        X, y        -- input, target
        k           -- number of folds
        folds       -- indices (see k_fold_indices)
        params      -- array containing a configuration (gamma, C, eps, tol, maxIter).
        threshold   -- error threshold for early stopping
        scaleY      -- boolean flag indicating wether scaling the targets or not.
    """
    err = 0
    M = None
    m = None
    stopped = False
    
    for i in range(k):
        if err > threshold:
            stopped = True
            break
        tr,tr_y, vs,vs_y = split_dataset(X,y, folds[i],folds[i+1])
        if(scaleY):
            tr_y, M, m = scale(tr_y)
        svr = SVR(gamma=params[0], C=params[1], eps=params[2], tol=params[3], maxIter=params[4])
        svr.fit(tr, tr_y)
        err += compute_error(svr, vs, vs_y, M, m)
        
    if stopped:
        print(f"Stopped early: Threshold= {threshold}, Err= {err}")
        err = -1
    else:
        err = err/k

    dump_svr_params("results_1.csv", (params[0], params[1], params[2], params[4], err))
    return err
        
