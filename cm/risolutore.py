#!/usr/bin/env python3
from myutils import *
import numpy as np
from qpsolvers import solve_qp

if __name__ == '__main__':
    args = get_cmdline_args()
    Q, q, _ = load_data(args.file, False)
    G, A, h, b = build_problem(len(q), 10)
    print(solve_qp(Q, q, G, h, A, b))