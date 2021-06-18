"""
This files contains some decorator functions we used in order to test the program.
"""

import time
import csv

def time_program(f):
    """
    Runs a function five times, prints the average time and return the result.
    """
    def wrapper(*args, **kwargs):
        times = []
        for i in range(5):
            before = time.time()
            ret = f(*args, **kwargs)
            elapsed = time.time() - before
            times.append(elapsed)
        
        avg = sum(times[1:])/len(times[1:])
        print(f"Time: {avg}")
        return ret, times
    return wrapper


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