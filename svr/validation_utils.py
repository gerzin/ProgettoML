import numpy as np
from SVR import SVR

'''
Estimate the error of the model
'''
def compute_error(svr, patterns, targets, M=None, m=None, err='mse'):
    outputs = [ svr.predict(x) for x in patterns ]
    if M and m:
        outputs = [ y*(M-m) + m for y in outputs ]
    
    if err == 'mse':
        e = np.square(outputs - targets).mean()
    
    return e


'''
Scale the values of the array between 0 and 1
'''
def scale(y):
    M, m = max(y), min(y)
    scaled = (y - m)/(M - m)
    return scaled, M, m


'''
Bring the scaled values back to the initial scale
'''
def scale_back(scaled, M, m):
    return scaled*(M - m) + m


'''
Return the indeces for dividing the folds
'''
def k_fold_indeces(n, k):
    fold_size = int(n/k)
    return [ i*fold_size for i in range(k) ] + [n]


'''
Split the dataset in training and validation set
according to the given interval
'''
def split_dataset(X,Y, start,end):
    vs_interval = [range(start,end)]
    
    tr = np.delete(X, vs_interval, 0)
    tr_y = np.delete(Y, vs_interval, 0)
    
    vs = X[start:end]
    vs_y = Y[start:end]
    
    return tr,tr_y, vs,vs_y


'''
'''
def k_fold_evaluate(X, y, k, folds, params, scaleY=False):
    err = 0
    M = None
    m = None
    for i in range(k):
        tr,tr_y, vs,vs_y = split_dataset(X,y, folds[i],folds[i+1])
        if(scaleY):
            tr_y, M, m = scale(tr_y)
        svr = SVR(gamma=params[0], C=params[1], eps=params[2], tol=params[3], maxIter=params[4])
        svr.fit(tr, tr_y)
        err += compute_error(svr, vs, vs_y, M, m)
    err = err/k
    return err
        