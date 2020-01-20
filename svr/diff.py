#!/usr/bin/env python3
from SLBQP import *
from myutils import *
import numpy as np
from qpsolvers import solve_qp

if __name__ == '__main__':
    args = get_cmdline_args()
    Q, q, _ = load_data(args.file, False)
    G, A, h, b = build_problem(len(q), 10)
    ALTRO = solve_qp(Q, q, G, h, A, b)
    
    r, NOSTRO, _ = SLBQP(Q, q, 10)
    print(r)
    print(np.linalg.norm(ALTRO-NOSTRO))
