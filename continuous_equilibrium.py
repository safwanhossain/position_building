import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
from pyomo.environ import *
from cost_models import get_price_vector
from tqdm import tqdm

def get_overall_optimal_welfare(game_dict):
    """ To verify, but I believe the optimal welfare setting can be thought of as a single buyer looking
    to buy \sum{V_i} position to minimize cost. And a single seller looking to maximize revenue.
    
    Notes: If we contrain the supply to be less than total demand or don't consider value of final position - opt welfare is 0
           If we constraint the supply to be less than toal demand, then opt welfare is 0, even with final position utiliyt
           If we don't constrain supply and consider final poisiton utility, it is beneficial for the supplier to over-supply and drive down cost for the
                 the buyer, knowing that at the end, it can build that position back cheaper since their temp impact beta is less than the beta faced by the buyers.
    """
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    v_total = sum(game_dict["Vs"])

    # Objective function
    def objective(strat):
        supply = strat[0:T]
        demand = strat[T:]
        demand.shape = (1, T)
        pts = get_price_vector(game_dict, demand, supply)
        last_step_walrus = pts[-1] - beta*(demand[0][T-1] - supply[T-1])
        revenue = np.dot(pts, supply) - np.sum(supply)*last_step_walrus
        print(f"Revenue: {revenue}")
        cost = np.dot(pts, demand[0])
        return -1*(revenue - cost)
    
    # Constraint: sum of demand == Vs[i]
    cons = ({
        'type': 'eq',
        'fun': lambda strat: np.sum(strat[T:]) - v_total
    },
    # {
    #     'type': 'ineq',
    #     'fun': lambda strat: v_total - np.sum(strat[0:T])
    # }
    )

    # Bounds: demand >= 0
    bounds = [(0, None) for _ in range(T*2)]

    # Initial guess: split Vs[i] evenly
    x0 = np.ones(T*2) * (v_total / T)
    result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)    
    return result.x[:T], result.x[T:], -1*result.fun


def get_seller_best_response(game_dict, trader_strat):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]

    # Objective function
    def objective(supply):
        pts = get_price_vector(game_dict, trader_strat, supply)
        revenue = np.dot(pts, supply)
        
        last_step_walrus = pts[-1] - beta*(np.sum(trader_strat[:,T-1]) - supply[T-1])
        return -1*revenue + np.sum(supply)*last_step_walrus
        #return -1*revenue
        #return -1*revenue + np.sum(supply)*last_step_walrus + beta*(np.sum(supply)**2)
    
    # Constraint: sum of demand == Vs[i]
    cons = ({
        'type': 'ineq',
        'fun': lambda demand: np.sum(Vs) - np.sum(demand)
    })

    # Bounds: demand >= 0
    bounds = [(0, None) for _ in range(T)]

    # Initial guess: split Vs[i] evenly
    x0 = np.ones(T) * (Vs[0] / T)
    result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=None)

    if not result.success:
        print("Scipy failed:", result.message)
        return None
    else:
        return result.x


def get_buyer_best_response(game_dict, trader_strat, supply_strat, i):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]

    # Objective function
    def objective(demand):
        pts = get_price_vector(game_dict, trader_strat, supply_strat, i, demand)
        cost = np.dot(pts, demand)
        return cost

    # Constraint: sum of demand == Vs[i]
    cons = ({
        'type': 'eq',
        'fun': lambda demand: np.sum(demand) - Vs[i]
    })

    # Bounds: demand >= 0
    bounds = [(0, None) for _ in range(T)]

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
   

def find_equilibrium_br(game_dict, supply_player=True, verbose=True, get_welfare=True):
    # Initialize supply
    n, T, Vs = game_dict["n"], game_dict["T"], game_dict["Vs"]
    supply = [0 for i in range(T)]

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

        if supply_player:
            br_supply = get_seller_best_response(game_dict, demand_matrix)
            step_size = np.linalg.norm(br_supply - supply) 
            if step_size >= eps:
                supply = br_supply
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

        print(f"The equilibrium supply is: {supply}\n") if verbose else None
        print(f"The equilibrium demand matrix is: {demand_matrix}\n") if verbose else None
        print(f"This leads to price {price_vector}") if verbose else None
        print(f"The cost to each trader is: {total_cost}") if verbose else None
        print(f"The total cost is: {sum(total_cost)}") if verbose else None
        print(f"The revenue is: {revenue}") if verbose else None

        if get_welfare:
            eq_welfare = revenue - sum(total_cost)
            supply_welf, demand_welf, opt_welfare = get_overall_optimal_welfare(game_dict) 
            print(f"\n\n The welfare of Equilibrium is: {eq_welfare}")
            print(f"The optimal welfare is: {opt_welfare} with supply: {supply_welf} and demand: {demand_welf}") 
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
            "beta" : beta
        }
        found, _, _ = find_equilibrium_br(game_dict, supply_player=True, verbose=False)
        supply_eq, demand_eq, opt_welfare = get_overall_optimal_welfare(game_dict)
        
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
        "alpha" : alpha/T
    }
    
    supply = [0, 0, 0]
    demand_matrix = np.array([
        [2, 2],
        [1, 1]
    ])

    br_supply = get_seller_best_response(game_dict, demand_matrix)
    price_vector, total_cost = get_cost(game_dict, demand_matrix, br_supply)
    revenue = np.dot(price_vector, br_supply)
    print(f"The demand is: {demand_matrix}")
    print(f"The br supply is: {br_supply}")
    print(f"The price is: {price_vector}")
    print(f"The revenue in br_supply is: {revenue}") 


if __name__ == "__main__":
    n, T, alpha, beta = 2, 2, 1, 1
    Vs = [10 for i in range(n)]
    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : 0,
        "Vs" : Vs,
        "alpha" : alpha/T,
        "beta" : beta
    }
    #best_response_test()
    find_equilibrium_br(game_dict, get_welfare=True, verbose=True)
    #check_random_equilibrium(3, 4, 1, 1)

    
    