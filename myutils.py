import argparse

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

