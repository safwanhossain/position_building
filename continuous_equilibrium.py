import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
from pyomo.environ import *
from cost_models import get_price_vector

def get_seller_best_response(game_dict, trader_strat):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]

    # Objective function
    def objective(supply):
        pts = get_price_vector(game_dict, trader_strat, supply)
        revenue = np.dot(pts, supply)
        return revenue*-1

    # Bounds: demand >= 0
    bounds = [(0, None) for _ in range(T)]

    # Initial guess: split Vs[i] evenly
    x0 = np.ones(T) * (Vs[0] / T)
    result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds)

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
   

def find_equilibrium_br():
    n, T, alpha, beta = 2, 3, 1, 1
    Vs = [10 for i in range(n)]
    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : 0,
        "Vs" : Vs,
        "alpha" : alpha/T,
        "beta" : beta
    }

    # Initialize supply
    supply = [0 for i in range(T)]

    # create an initial demand matrix where everyone buys the whole order up-front.
    demand_matrix = np.zeros((n, T))
    for i in range(n):
        demand_matrix[i, 0] = Vs[i]
    
    # Now try and find a Nash Equilibrium through best-response play.
    iter, eps, supply_player = 0, 0.01, True
    while True:
        update = False
        for i in range(n):
            br_demand_i = get_buyer_best_response(game_dict, demand_matrix, supply, i)
            if np.linalg.norm(br_demand_i - demand_matrix[i]) >= eps:
                demand_matrix[i] = br_demand_i
                update = True
        
        if supply_player:
            br_supply = get_seller_best_response(game_dict, demand_matrix)
            if np.linalg.norm(br_supply - supply) >= eps:
                supply = br_supply
                update = True
        iter += 1
        
        print(f"Completed iter: {iter}")
        if not update:
            print("Found Equilibrium")
            break

    price_vector, total_cost = get_cost(game_dict, demand_matrix, br_supply)
    revenue = np.dot(price_vector, br_supply)

    print(f"The equilibrium supply is: {br_supply}")
    print(f"The equilibrium demand matrix is: {demand_matrix}\n\n")
    print(f"This leads to price {price_vector}")
    print(f"The cost to each trader is: {total_cost}")
    print(f"The total cost is: {sum(total_cost)}")
    print(f"The revenue is: {revenue}")


def best_response_test():
    n, T, alpha, beta = 2, 3, 1, 1
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
        [0.476, 4.76, 4.76],
        [5.23, 2.72, 2.04]
    ])

    br_supply = get_seller_best_response(game_dict, demand_matrix)
    price_vector, total_cost = get_cost(game_dict, demand_matrix, br_supply)
    revenue = np.dot(price_vector, br_supply)
    print(f"The demand is: {demand_matrix}")
    print(f"The br supply is: {br_supply}")
    print(f"The price is: {price_vector}")
    print(f"The revenue in br_supply is: {revenue}") 


if __name__ == "__main__":
    #best_response_test()
    find_equilibrium_br()
    