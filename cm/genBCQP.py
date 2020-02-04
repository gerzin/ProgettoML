#!/usr/bin/env python3
import numpy as np
import numpy.linalg as la
import random

def genBCQP( n , actv=0.5 , rank=1.1 , ecc=0.99 , seed=0, u=10):
	Q, q = None, None

	G = np.random.rand(round(rank * n) , n )
	Q = np.transpose(G) @ G

	z = np.zeros(n)
	outb = [ i <= actv for i in np.random.rand(n)]
	
	lr = [i <= 0.5 for i in np.random.rand(n)]
	
	l = [a and b for (a,b) in zip(outb, lr)]
	r = [a and not b for (a,b) in zip(outb, lr)]
	

	for (i,j) in enumerate(l):
		if j:
			z[i] = -u*random.random()

	for (i,j) in enumerate(r):
		if j:
			z[i] = u*(1 + random.random())

	outb = [not i for i in outb]
	for (i,j) in enumerate(outb):
		if j:
			z[i] = random.random()*u

	q = -Q @ z

	return Q, q, u

if __name__ == "__main__":
	Q,q,u = genBCQP(4)
	print(Q)
	print(q)
	print(u)
	

