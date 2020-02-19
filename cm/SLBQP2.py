import numpy as np
from myutils import print_decreasing 

def SLBQP(Q, q, l, u, eps=1e-6, maxIter=5000, m1=0.0001, astart=1, tau=0.9, verbose=False): 

    n = len(q)
    x = np.zeros(n)
    active_indeces1 = [False for i in range(n)]
    active_indeces2 = [False for i in range(n)]
    
    ite = 1
    
    OLD = None
    
    if verbose:
        print("Iter.\tFunction val\t||gradient||\t||direction||\talpha")

    while True:        
        # Compute function value (v), gradient (g) and descent direction (d)
        Qx = Q @ x
        v = (0.5)*(x @ Qx) + (q @ x) #+ 0.1*np.linalg.norm(x, ord=1)
        g = Qx + q
        
        d = -g
        
        cont = 2*n
        d_sum = 0
        for i in range(n):
            active_indeces1[i] = (x1[i] == l) or (x1[i] == u)
            if(active_indeces1[i]):
                d1[i] = 0
                cont -= 1
            d_sum += d1[i]
            
        for i in range(n):
            active_indeces2[i] = (x2[i] == l) or (x2[i] == u)
            if(active_indeces1[i]):
                d2[i] = 0
                cont -= 1
            d_sum += d2[i]
        
        temp = d_sum/cont
        for i in range(n):
            if not active_indeces1[i]:
                d1[i] = d1[i] - temp
        for i in range(n):
            if not active_indeces2[i]:
                d2[i] = d2[i] - temp

        
        d_norm = np.linalg.norm(d)
        g_norm = np.linalg.norm(g)
        if verbose:
            #print("%5d\t%1.16e\t%1.16e\t%1.16e" % (ite, v, g_norm, d_norm), end="")
            OLD = print_decreasing(ite, v, g_norm, d_norm, OLD)
            print("")
        
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
                    
#        phip0 = - d_norm*d_norm
#        alpha = astart
#        
#        p = x + alpha*d
#        va = (0.5)*(p @ (Q @ p)) + (q @ p) + 0.1*np.linalg.norm(p, ord=1)
#        iterazioni = 1
#        while(va > v + m1*alpha*phip0):
#            iterazioni += 1
#            alpha = tau*alpha
#            p = x + alpha*d
#            va = (0.5)*(p @ (Q @ p)) + (q @ p) + 0.1*np.linalg.norm(p, ord=1)
#        
#        print
#        print("\t%5d\t%1.16e" % (iterazioni,alpha), end="")
#        
#        alpha = min(max_alpha, alpha)

        #print("\t%1.16e" % (alpha))
        # Compute next iterate
        x = x + alpha * d
        
        ite = ite + 1