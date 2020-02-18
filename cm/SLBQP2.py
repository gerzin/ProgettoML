import numpy as np

def SLBQP(Q, q, l, u, x=None, eps=1e-6, maxIter=5000, verbose=False): 

    n = len(q)
    x = np.zeros(n)
    active_indeces = [False for i in range(n)]
    
    ite = 1
    
    if verbose:
        print("Iter.\tFunction val\t||gradient||\t||direction||\talpha")

    while True:        
        # Compute function value (v), gradient (g) and descent direction (d)
        Qx = Q @ x
        v = (0.5)*(x @ Qx) + (q @ x)
        g = Qx + q
        d = -g
        
        for i in range(n):
            #active_indeces[i] = (x[i] == l and d[i] < 0) or (x[i] == u and d[i] > 0)
            active_indeces[i] = (x[i] == l) or (x[i] == u)
            if(active_indeces[i]):
                d[i] = 0

        cont = n - sum(active_indeces)
        temp = sum(d)/cont
        
        #print(active_indeces)
        for i in range(n):
            if not active_indeces[i]:
                #print(f"{x[i]},{d[i]}")
                d[i] = d[i] - temp

        
        d_norm = np.linalg.norm(d)
        g_norm = np.linalg.norm(g)
        if verbose:
            print("%5d\t%1.16e\t%1.16e\t%1.16e" % (ite, v, g_norm, d_norm), end="")
        
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
                if(max_alpha==0):
                    print(f"\n{j}{x[j]},{d[j]}")
                    return
            elif(d[j] < 0):
                max_alpha = min( max_alpha, (l - x[j])/d[j] )
                if(max_alpha==0):
                    print(f"\n{j},{x[j]},{d[j]}")
                    return

        # Exact line search toward the minimum
        quad = np.dot(d, np.dot(Q, d))
        if(quad <= 1e-16):
            # If the quadratic part is zero, take the maximum stepsize
            alpha = max_alpha
        else:
            # Otherwise select the minimum between the optimal unbounded
            # stepsize and the maximum feasible stepsize
            alpha = min(max_alpha, (d_norm**2)/quad)

        print("\t%1.16e" % (alpha))
        # Compute next iterate
        x = x + alpha * d
        
        ite = ite + 1