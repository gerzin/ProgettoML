import numpy as np
import itertools
from numba import njit, prange
import time

def time_program(f):
    def wrapper(*args, **kwargs):
        times = []
        for i in range(5):
            before = time.time()
            ret = f(*args, **kwargs)
            elapsed = time.time() - before
            times.append(elapsed)
        
        avg = sum(times[1:])/len(times[1:])
        print(f"Time: {avg}")
        return ret
    return wrapper

#@njit
def custom_clip(a, min, max):
    for i in prange(len(a)):
        if a[i] < min:
            a[i] = min
        elif a[i] > max:
            a[i] = max
    return a


#@njit()
def compute_x_r(d1, d2, lmb, u):
    """Compute the optimal value for x given lambda.

    Params:
        d1  -- first block of the direction vector
        d2  -- second block of the direction vector
        lmb -- current value for lambda
        u   -- upper bound of the feasible region

    Returns:
        x1  -- first block of the optimal x
        x2  -- second block of the optimal x
        r   -- value of r given the optimal x and lambda
    """

    # x = d + lmb * a
    # with a = [1...1,-1...-1]
    x1 = d1 + lmb
    x2 = d2 - lmb

    # 'Apply' the box constraints
    x1 = np.clip(x1, 0, u)
    x2 = np.clip(x2, 0, u)
    
    #x1 = custom_clip(x1, 0, u)
    #x2 = custom_clip(x2, 0, u)
    
    # r = a'x
    # with a = [1...1,-1...-1]
    r = np.sum(x1) - np.sum(x2)

    return x1, x2, r


def project_Goldstein(d1, d2, u, lmb, d_lmb, eps):
    """ Return the projection of the point d over the feasible region
    defined by 0 <= x <= u and ax = 0 with a = [1...1,-1...-1]

    Params:
        d1      -- first block of the direction vector
        d2      -- second block of the direction vector
        u       -- upper bound of the feasible region
        lmb     -- 
        d_lmb   -- delta lambda
        eps     -- precision

    Returns:
        x1      -- first block of the projected point
        x2      -- second block of the projected point
    """

    # BRACKETING PHASE -----
    # ----------------------

    # Compute x and r and check whether it found the minimum
    x1, x2, r = compute_x_r(d1, d2, lmb, u)
    if abs(r) < eps:
        return x1, x2

    if r < 0:
        # r(λ) < 0 -> search for a λ | r(λ) > 0

        # initialize lower bounds and update
        lmb_l = lmb
        r_l = r
        lmb += d_lmb
        x1, x2, r = compute_x_r(d1, d2, lmb, u)
        if abs(r) < eps:
            return x1, x2

        while r < 0:
            # update lower bounds and lambda
            lmb_l = lmb
            r_l = r
            s = max(r_l/r - 1, 0.1)
            d_lmb += d_lmb/s
            lmb += d_lmb

            # compute x and r and check whether it found the minimum
            x1, x2, r = compute_x_r(d1, d2, lmb, u)
            if abs(r) < eps:
                return x1, x2

        # initialize upper bounds
        lmb_u = lmb
        r_u = r

    else:
        # r(λ) > 0 -> search for a λ' | r(λ') < 0

        # initialize upper bounds and update lambda
        lmb_u = lmb
        r_u = r
        lmb -= d_lmb

        # Compute x and r and check whether it found the minimum
        x1, x2, r = compute_x_r(d1, d2, lmb, u)
        if abs(r) < eps:
            return x1, x2

        while r > 0:
            # update upper bounds and lambda
            lmb_u = lmb
            r_u = r
            s = max(r_u/r - 1, 0.1)
            d_lmb += d_lmb/s
            lmb -= d_lmb

            # compute x and r and check whether it found the minimum
            x1, x2, r = compute_x_r(d1, d2, lmb, u)
            if abs(r) < eps:
                return x1, x2

        # initialize lower bounds
        lmb_l = lmb
        r_l = r

    # secant phase
    s = 1 - r_l/r_u
    d_lmb = d_lmb/s
    lmb = lmb_u - d_lmb
    x1, x2, r = compute_x_r(d1, d2, lmb, u)

    while(abs(r) >= eps):
        if(r > 0):
            # move upper bound
            if(s <= 2):
                lmb_u = lmb
                r_u = r
                s = 1 - r_l/r_u
                d_lmb = (lmb_u - lmb_l)/s
                lmb = lmb_u - d_lmb
            else:
                s = max(r_u/r - 1, 0.1)
                d_lmb = (lmb_u - lmb)/s
                lmb_new = max(lmb - d_lmb, 0.75*lmb_l + 0.25*lmb)
                lmb_u = lmb
                r_u = r
                lmb = lmb_new
                s = (lmb_u - lmb_l)/(lmb_u - lmb)
        else:
            # move lower bound
            if(s >= 2):
                lmb_l = lmb
                r_l = r
                s = 1 - r_l/r_u
                d_lmb = (lmb_u - lmb_l)/s
                lmb = lmb_u - d_lmb
            else:
                s = max(r_l/r - 1, 0.1)
                d_lmb = (lmb - lmb_l)/s
                lmb_new = min(lmb + d_lmb, 0.75*lmb_u + 0.25*lmb)
                lmb_l = lmb
                r_l = r
                lmb = lmb_new
                s = (lmb_u - lmb_l)/(lmb_u - lmb)

        x1, x2, r = compute_x_r(d1, d2, lmb, u)
    return x1, x2


# @njit
def project_Rosen(d1, d2, x1, x2, u):
    """ Rosen projection of d over the feasible region 0 <= x <= u

    Params:
        d1  -- first block of the direction vector
        d2  -- second block of the direction vector
        x1  -- first block of the iterate
        x2  -- second block of the iterate
        u   -- upper bound of the feasible region

    Returns:
        proj1  -- first block of the projected gradient
        proj2  -- second block of the projected gradient
    """

    n = len(x1)

    eps = 1e-12
    _u = u - eps

    # Active components masks
    active_indeces1 = project_Rosen._active_indeces1
    active_indeces2 = project_Rosen._active_indeces2
    #active_indeces1 = [(x1[i] < eps and d1[i] < 0) or (x1[i] > _u and d1[i] > 0) for i in range(n)]
    #active_indeces2 = [(x2[i] < eps and d2[i] < 0) or (x2[i] > _u and d2[i] > 0) for i in range(n)]
    active_indeces = np.concatenate((active_indeces1, active_indeces2), axis=None)

    n_active1 = sum(active_indeces1)
    n_active2 = sum(active_indeces2)

    # Free components masks
    free_indeces1 = [not i for i in active_indeces1]
    free_indeces2 = [not i for i in active_indeces2]

    # Number of free components
    f = sum(free_indeces1) + sum(free_indeces2)

    # Sum of positive and negative free components of d
    sum_pos = np.sum(d1[free_indeces1])
    sum_neg = np.sum(d2[free_indeces2])

    # Projection
    proj1 = np.zeros(n)
    proj2 = np.zeros(n)

    d = np.concatenate((d1, d2), axis=None)
    changed = True
    count = 0
    while(changed):
        if f == 0:
            return proj1, proj2, count
            
        changed = False
        count += 1
        #print(active_indeces)

        # Compute the Lagrange multipliers
        Ak = np.array(
            [1 if v > _u else -1 for v in itertools.chain(x1[active_indeces1], x2[active_indeces2])])
        bk = np.copy(Ak)
        bk[n_active1:] = np.negative(bk[n_active1:])
        mu = Ak*d[active_indeces] - bk/f * sum_pos + bk/f * sum_neg

        # Check if the Lagrange multipliers are all positive
        # In case one is negative, the corresponding constraint is removed from the active set
        k = 0
        for i in range(n):
            if active_indeces1[i]:
                if mu[k] < 0:
                    #print("\t\t\tmu[k]<0")
                    active_indeces1[i] = False
                    active_indeces[i] = False
                    n_active1 -= 1

                    sum_pos += d1[i]

                    free_indeces1[i] = True
                    f += 1

                    changed = True
                    break
                else:
                    k += 1

        if changed:
            continue

        for i in range(n):
            if active_indeces2[i]:
                if mu[k] < 0:
                    #print("\t\t\tmu[k]<0")
                    active_indeces2[i] = False
                    active_indeces[n+i] = False
                    n_active2 -= 1

                    sum_neg += d2[i]

                    free_indeces2[i] = True
                    f += 1

                    changed = True
                    break
                else:
                    k += 1

        if changed:
            continue
        # - - - - - - - - - - - - - - - - - - - -

        # Compute the projection
        if(f == 0):
            return proj1, proj2, count
        else:
            v = (sum_pos - sum_neg) / f

        proj1[free_indeces1] = d1[free_indeces1] - v
        proj2[free_indeces2] = d2[free_indeces2] + v

        # Check if the projection does not point outside the feasible region
        # In case one component does, add it to the active set
        for i in range(n):
            if (x1[i] < eps and proj1[i] < 0) or (x1[i] > _u and proj1[i] > 0):
                #print("\t\t\tproj[i] wrong")
                active_indeces1[i] = True
                active_indeces[i] = True
                n_active1 += 1

                sum_pos -= d1[i]

                free_indeces1[i] = False
                f -= 1

                proj1[i] = 0

                changed = True
                break

        if (changed):
            continue

        for i in range(n):
            if (x2[i] < eps and proj2[i] < 0) or (x2[i] > _u and proj2[i] > 0):
                #print("\t\t\tproj[i] wrong")
                active_indeces2[i] = True
                active_indeces[n+i] = True
                n_active2 += 1

                sum_neg -= d2[i]

                free_indeces2[i] = False
                f -= 1

                proj2[i] = 0

                changed = True
                break
        # - - - - - - - - - - - - - - - - - - - -

    return proj1, proj2, count


if __name__ == '__main__':
    # test the projections here
    pass
