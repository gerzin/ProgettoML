#!/usr/bin/env python3
from svr.myutils import *
import numpy as np
if __name__ == '__main__':
	args = get_cmdline_args()
	data = load_data(args.file, delfirst=False, split=False)
	A, B = splitHorizontally(data, float(args.percentage))
	np.savetxt("training.csv", A, fmt="%.18f" , delimiter=",")
	np.savetxt("test.csv", B, fmt="%.18f" , delimiter=",")