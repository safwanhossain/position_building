import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
from pyomo.environ import *
from cost_models import get_price_vector
from tqdm import tqdm


def get_optimal_welfare(game_dict, supply):
    """ To verify, but I believe the optimal welfare setting can be thought of as a single buyer looking
    to buy \sum{V_i} position to minimize cost. And a single seller looking to maximize revenue.
    
    Notes: If we contrain the supply to be less than total demand or don't consider value of final position - opt welfare is 0
           If we constraint the supply to be less than toal demand, then opt welfare is 0, even with final position utiliyt
           If we don't constrain supply and consider final poisiton utility, it is beneficial for the supplier to over-supply and drive down cost for the
                 the buyer, knowing that at the end, it can build that position back cheaper since their temp impact beta is less than the beta faced by the buyers.
    """
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    
    # Objective function
    def objective(demand):
        demand.shape = (n, T)
        pts = get_price_vector(game_dict, demand, supply)
        total_utility = 0
        # TODO: We can vectorize these below if runtime becomes an issue here
        if reserve:
            total_utility = np.sum([reserve[i]*np.sum(demand[i]) - np.dot(pts, demand[i]) for i in range(n)])
        else:
            total_utility = -1*np.sum([np.dot(pts, demand[i]) for i in range(n)])
            
        # scipy default is to minimize - hence the negative
        return -1*total_utility
    
    # Constraint: sum of demand == Vs[i] when no reserve; demand <= Vs[i] with reserve
    cons = []
    for i in range(n):
        cons.append({
            'type': 'eq' if not reserve else 'ineq',
            'fun': lambda demand, i=i: Vs[i] - np.sum(demand[i*T:(i+1)*T])
        })
    
    # No per-round bounds
    bounds = [(None, None) for _ in range(n*T)]

    x0 = np.ones(n*T)
    result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)        
    
    # If we are not using reserve, then return the positive cost (lower better)
    # Otherwise, return the utility (higher better)
    return result.x.reshape((n,T)), result.fun if not reserve else -1*result.fun
    
        

def get_buyer_best_response(game_dict, trader_strat, supply_strat, i):
    T = game_dict["T"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]

    # Objective function
    def objective(demand):
        pts = get_price_vector(game_dict, trader_strat, supply_strat, i, demand)
    
        total_utility = 0
        if reserve:
            total_utility = np.sum(demand)*reserve[i] - np.dot(pts, demand)
        else:
            total_utility = -1*np.dot(pts, demand) 
            
        # scipy default is to minimize - hence the negative
        return -1*total_utility

    # Constraint: sum of demand == Vs[i]
    cons = ({
        'type': 'eq' if not reserve else 'ineq',
        'fun': lambda demand: Vs[i] - np.sum(demand)
    })

    # Bounds: demand >= 0
    bounds = [(None, None) for _ in range(T)]

    # Initial guess: split Vs[i] evenly
    x0 = np.ones(T) * (Vs[i] / T)
    result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

    if not result.success:
        print("Scipy failed:", result.message)
        return None
    else:
        return result.x


def get_cost(game_dict, demand_matrix, supply):
    price_vector = get_price_vector(game_dict, demand_matrix, supply)
    total_costs = []
    for i in range(game_dict["n"]):
        cost = np.dot(price_vector, demand_matrix[i])
        total_costs.append(cost) 
    return price_vector, total_costs
   

def verify_equilibrium(game_dict, demand_matrix, supply):
    """ Verify whether a given set of strategies at at an equilibrium
    """
    n, T, Vs = game_dict["n"], game_dict["T"], game_dict["Vs"]

    price_vector, total_cost = get_cost(game_dict, demand_matrix, supply)
    print(f"The total cost of current strategy is: {total_cost}")

    # check the best response for each of the buyers
    eps = 0.01
    for i in range(n):
        br_demand_i = get_buyer_best_response(game_dict, demand_matrix, supply, i)
        step_size = np.linalg.norm(br_demand_i - demand_matrix[i]) 
        if step_size >= eps:
            print(f"Agent {i} best responds with: {br_demand_i}")
            return False

    return True


def find_equilibrium_br(game_dict, supply, verbose=True, get_welfare=True):
    # Initialize supply
    n, T, Vs = game_dict["n"], game_dict["T"], game_dict["Vs"]
    reserve = game_dict["reserve"]

    # create an initial demand matrix where everyone buys the whole order up-front.
    demand_matrix = np.zeros((n, T))
    for i in range(n):
        demand_matrix[i, 0] = Vs[i]
    
    # Now try and find a Nash Equilibrium through best-response play.
    iter, eps, max_iter = 0, 0.01, 1000
    while True:
        update = False
        step_sizes = []

        for i in range(n):
            br_demand_i = get_buyer_best_response(game_dict, demand_matrix, supply, i)
            step_size = np.linalg.norm(br_demand_i - demand_matrix[i]) 
            if step_size >= eps:
                demand_matrix[i] = br_demand_i
                update = True
            step_sizes.append(step_size)

        iter += 1
        print(f"Iter: {iter} with the largest step size being: {max(step_sizes)}") if verbose else None
        
        if not update:
            found = True
            print(f"Found Equilibrium in {iter} iterations") if verbose else None
            break
        if iter >= 1000:
            found = False
            break

    if found:
        price_vector, total_cost = get_cost(game_dict, demand_matrix, supply)
        revenue = np.dot(price_vector, supply)

        print(f"The equilibrium demand matrix is: {demand_matrix}") if verbose else None
        print(f"This leads to price {price_vector}") if verbose else None
        print(f"The cost to each trader is: {total_cost}") if verbose else None
        print(f"The total cost is: {sum(total_cost)}") if verbose else None
    
        if get_welfare:
            if reserve:
                eq_welfare = np.sum([reserve[i]*np.sum(demand_matrix[i]) - np.dot(price_vector, demand_matrix[i]) for i in range(n)])
            else:
                eq_welfare = sum(total_cost)
            demand_welf, opt_welfare = get_optimal_welfare(game_dict, supply)
            price_opt_welfare, _ = get_cost(game_dict, demand_welf, supply) 
            print(f"\n\n The welfare of Equilibrium is: {eq_welfare}")
            print(f"The optimal welfare is: {opt_welfare} with demand: {demand_welf} and prices: {price_opt_welfare}") 
    else:
        print(f"Equilibrium not found in {max_iter} iterations.")
        print(f"Demand matrix: {demand_matrix}")
        print(f"Supply vector: {supply}")

    return found, demand_matrix, supply


def check_random_equilibrium(n, T, alpha, beta):
    num_iters = 100
    for i in tqdm(range(num_iters)):
        Vs = np.random.randint(0, 20, n)
        game_dict = {
            "n" :   n,
            "T" :   T,
            "p_0" : 0,
            "Vs" : Vs,
            "alpha" : alpha/T,
            "beta" : beta,
            "reserve" : None
        }
        found, _, _ = find_equilibrium_br(game_dict, supply_player=True, verbose=False)
        supply_eq, demand_eq, opt_welfare = get_optimal_welfare(game_dict)
        
        if not found:
            print(Vs)
            exit(0)


def best_response_test():
    n, T, alpha, beta = 2, 2, 1, 1
    Vs = [10 for i in range(n)]
    game_dict = {
        "n" :   n,
        "T" :   T,
        "beta" : beta,
        "p_0" : 0,
        "Vs" : Vs,
        "alpha" : alpha/T,
        "reserve" : [5, 5]
    }
    
    supply = [0, 0, 0]
    demand_matrix = np.array([
        [5, 5],
        [1, 9]
    ])
    best_response = get_buyer_best_response(game_dict, demand_matrix, supply, 0)
    print(f"Player 0 best response is: {best_response}")


if __name__ == "__main__":
    n, T, alpha, beta = 3, 10, 3, 1
    Vs = [5,10, 15]
    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : 5,
        "Vs" : Vs,
        "alpha" : alpha/T,
        "beta" : beta,
        "reserve" : None
    }
    supply = [1 for i in range(T)]
    #demand, obj = get_overall_optimal_welfare(game_dict, supply)
    #print(f"The demand is: {demand}, obj is: {obj}")
    
    #best_response_test()
    find_equilibrium_br(game_dict, supply, get_welfare=True, verbose=True)
    #check_random_equilibrium(3, 4, 1, 1)

    
    