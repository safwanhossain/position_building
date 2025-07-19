import numpy as np
import osqp
from scipy.sparse import csc_matrix
import sys

import time

def get_linear_operator(game_dict):
    """ Recall that our equilibrium can be expressed as a variational inequality
    with a linear operator Mx + c. This operator matrix depends only on the instance params
    alpha and b depends on supply, reserve, and initial price. 
    
    For large instances, these are large and they should only be computed once and stored in mem.
    """
    n, T, alpha, beta = game_dict["n"], game_dict["T"], game_dict["alpha"], game_dict["beta"] 
    p_0 = game_dict["p_0"]
    reserve = np.array(game_dict["reserve"])
    supply = np.array(game_dict["supply"])

    # construct the M matrix
    Q = np.diag([alpha + 2*beta for i in range(T)]) + alpha*np.ones((T,T))
    A = alpha*np.tril(np.ones((T, T))) + np.diag([beta for i in range(T)])
    I_n = np.diag([1 for i in range(n)])
    J_n = np.ones((n,n))
    M = np.kron(J_n, A) - np.kron(I_n, A) + np.kron(I_n, Q)

    # construct the c vector
    assert reserve is not None
    B = -alpha*np.tril(np.ones((T, T))) - beta*np.eye(T)
    supp = np.matmul(B, supply)
    supp_kron = np.kron(np.ones(n), supp)
    r_p = [p_0 - reserve[i] for i in range(n)]
    r_p_kron = np.kron(r_p, np.ones(T)) 
    c = supp_kron + r_p_kron
    return M, c


def project_feasible_analytical(game_dict, z):
    n, T = game_dict["n"], game_dict["T"]
    Vs = game_dict["Vs"]
    H = z.reshape((n,T))
 
    offset = np.maximum(np.sum(H, axis=1) - Vs, np.zeros(n)) / T
    H -= offset[:, None]
    return H.flatten()


def project_feasible(game_dict, z):
    """ Given z, which is an nT dim array of demands, project this to the feasibly region.
    Note that the feasible region must be convex, which means this projection is only well-defined
    if we have inequality constraints. For equality constraints, we can solve directly using matrix
    inverse, or with the inequality constraints but with very large reserve prices.

    Projection under l2 distance is generally quadratic program: we want to project z to a convex region
    minimize 1/2||x - z||_2^2 = minimize 1/2 x^T I x - z^T x. The constraints can be expressed as l <= Ax <= c
    In our case, we want projection such that each T sized vector sums to less than V_i, with no lower bound 
    contraints for now. We will used OSQP, which is an extremely fast solver specifically for quadratic programs.
    
    IMPORTANT: USE THIS FUNCTION IF YOU HAVE CONSTRAINTS LIKE L <= AX <= C AND h_1 <= x <= h_2. If you only
    have constraints like L <= AX <= C (our current setting), the projection can be analytically computed 
    and will be much faster to use project_feasible_analytical
    """
    n, T = game_dict["n"], game_dict["T"]
    reserve = game_dict["reserve"]
    assert reserve is not None
    Vs = game_dict["Vs"]

    # quadratic objective
    I = csc_matrix(np.eye(n*T))
    q = -1*np.array(z)

    # leq sum constraints
    I_n = np.eye(n)
    A = np.kron(I_n, np.ones(T))
    A = csc_matrix(A)
    c = np.array(Vs)
    l = -np.inf * np.ones(n)

    prob = osqp.OSQP()
    prob.setup(P=I, q=q, A=A, l=l, u=c, verbose=False)
    res = prob.solve()
    return res.x


def extra_gradient_equilibrium(game_dict, eta=None):
    """ Express the equilibrium solution as a joint variational inequality and use the projected extra gradient
    algorithm with step size eta to solve this. At every step, we do a projected look ahead, and the update the
    current value based on the gradient direction from the projected lookahead.
    
    For linear convergence, we need the step size eta such that
    \eta <= 1/L, where L = nTa+a+b(n+1).
    Note this is different than one in paper - that is a more conservative value used for easier convergence proof.
    """
    eps = 0.0001
    n, T, Vs = game_dict["n"], game_dict["T"], game_dict["Vs"]
    M, b = get_linear_operator(game_dict)
    initial_guess = np.concatenate([[Vs[i]/T for t in range(T)] for i in range(n)])
    L = (n*T + 1)*alpha + (n+1)*beta
    if eta is None:
        eta = 0.9/L
    else:
        assert eta <= 1/L
        
    prev_x, curr_x = np.zeros(n*T), initial_guess
    while np.linalg.norm(prev_x-curr_x) >= eps:
        prev_x = curr_x.copy()
        f_prev = np.matmul(M, prev_x) + b
        lookahead_x = prev_x - eta*f_prev
        lookahead_x = project_feasible_analytical(game_dict, lookahead_x)
        
        f_lookahead = np.matmul(M, lookahead_x) + b
        curr_x = prev_x - eta*f_lookahead
        curr_x = project_feasible_analytical(game_dict, curr_x)
    
    equi_demand = curr_x.reshape(n, T)
    return equi_demand
     

if __name__ == "__main__":
    n, T, alpha, beta = 15, 500, 2, 1
    Vs = [20 for i in range(n)]
    reserve = [5 for i in range(n)]
    supply = [0 for i in range(T)]

    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : 2.0,
        "Vs" : Vs,
        "alpha" : alpha,
        "beta" : beta,
        "supply" : supply,
        "reserve" : reserve,
        "exp" : 1
    }

    #proj = project_feasible(game_dict, [10, 10, 10, 10, 10, 10, 10, 10])
    start = time.time()
    out = extra_gradient_equilibrium(game_dict)
    end = time.time()
    print(f"Took: {end - start} seconds")
    print(out)

