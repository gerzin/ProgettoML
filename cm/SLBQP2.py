import numpy as np

def SLBQP(Q, q, u, a, x=None, eps=1e-6, maxIter=1000, verbose=False, stopAtIter=False): 

    n = len(q)
    active_indeces = np.zeros(n)
    x = np.zeros(n)    
    
    ite = 1
    
    if verbose:
        print("Iter.\tFunction val\t||gradient||\t||direction||\t  |Z|\t  |U|\tStepsize")

    while True:        
        # Compute function value (v), gradient (g) and descent direction (d)
        Qx = Q @ x
        v = (0.5)*(x @ Qx) + (q @ x)
        g = Qx + q
        d = -g
        
        cont = n
        for i in range(n):
            if(x[i] == 0 and d[i] < 0) or (x[i] == u and d[i] > 0):
                cont -= 1
                active_indeces[i] = 1
                
        n2 = int(n/2)
        temp = np.full((n2, n2), 1/cont)
        B = np.block([[temp, -temp], [-temp, temp]])
        for i in range(n):
            if(active_indeces[i] == 1):                    
                B[0:n, i] = 0
                B[i, 0:n] = 0
                B[i][i] = 1
        
        print(cont)
        
        d = (np.eye(n) - B) @ d  
        
        d_norm = np.linalg.norm(d)
        
        if(d_norm < eps):
            if verbose :
                print("")
            return ('optimal', x, v)
        if(maxIter != -1 and ite >= maxIter):
            if verbose:
                print("")
            return ('terminated', x, v)
        
        # Compute the maximum feasible stepsize
        max_alpha = np.Inf
        for j in range(len(d)):
            if(d[j] > 0):
                max_alpha = min( max_alpha, (u - x[j])/d[j] )
            elif(d[j] < 0):
                max_alpha = min( max_alpha, (-x[j])/d[j] )

        # Exact line search toward the minimum
        quad = np.dot(d, np.dot(Q, d))
        if(quad <= 1e-16):
            # If the quadratic part is zero, take the maximum stepsize
            alpha = max_alpha
        else:
            # Otherwise select the minimum between the optimal unbounded
            # stepsize and the maximum feasible stepsize
            alpha = min(max_alpha, (d_norm**2)/quad)

        # Compute next iterate
        print(alpha)
        x = x + alpha * d
        
        ite = ite + 1
        
        #input(">")