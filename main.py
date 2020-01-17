#!/usr/bin/env python3
from myutils import *
import numpy as np
from SLBQP import *

if __name__ == '__main__':
    args = get_cmdline_args()
    X, Y1, Y2 = load_data(args.file, False)
    G, a, h, b = build_problem(len(Y1), 10)
    _, x, e = SLBQP(X, Y1, 10)
