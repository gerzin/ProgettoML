#!/usr/bin/env python3
from myutils import *
import numpy as np

if __name__ == '__main__':
    args = get_cmdline_args()
    X, Y1, Y2 = load_data(args.file)
    print(Y1)
