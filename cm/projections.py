import numpy as np


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

            # Compute x and r and check whether it found the minimum
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

            # Compute x and r and check whether it found the minimum
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

            # Compute x and r and check whether it found the minimum
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

            # Compute x and r and check whether it found the minimum
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


if __name__ == '__main__':
    # test the projections here
    pass
