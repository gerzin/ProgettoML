import argparse
import numpy as np

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
