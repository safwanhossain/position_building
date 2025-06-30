import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
from pyomo.environ import *
from cost_models import walruss_model

def get_trader_best_response_scipy(game_dict, trader_strat, supply_strat, i, symmetric=False):
    n, T = game_dict["n"], game_dict["T"]
    alpha, p_0 = game_dict["alpha"], game_dict["p_0"]
    Vs = game_dict["Vs"]

    # Objective function
    def objective(demand):
        obj = 0
        for t in range(T):
            if symmetric:
                obj += p_0 * demand[t]
                for l in range(t+1):
                    obj += n*alpha*demand[t]*demand[l]
                obj -= alpha*demand[t]*np.sum(supply_strat[:t+1]) 
            else:
                obj += p_0 * demand[t]
                for l in range(t+1):
                    total_demand = np.sum(trader_strat[:, l]) - trader_strat[i, l] + demand[l]
                    obj += alpha * (total_demand - supply_strat[l]) * demand[t]
        return obj

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
        print("Success")
        print("Optimal objective value:", result.fun)
        return result.x


def get_trader_best_response_pyomo(game_dict, trader_strat, supply_strat, i, symmetric=False):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta,  p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]

    model = ConcreteModel()
    model.T = RangeSet(0, T-1)

    # Variables
    model.demand = Var(model.T, domain=NonNegativeReals)

    # Demand sum constraint
    def demand_sum_rule(m):
        return sum(m.demand[t] for t in m.T) == Vs[i]
    model.demand_sum_con = Constraint(rule=demand_sum_rule)

    # Objective
    def obj_rule(m):
        obj = 0
        for t in m.T:
            if symmetric:
                obj += p_0 * m.demand[t]
                for l in range(t+1):
                    obj += n*alpha*m.demand[t]*m.demand[l]
                obj -= alpha*m.demand[t]*np.sum(supply_strat[:t+1]) 
            else:
                obj += p_0 * m.demand[t]
                for l in range(t+1):
                    obj += alpha * m.demand[t] * m.demand[l]
                    remaining_demand = sum(trader_strat[:, l]) - trader_strat[i, l]
                    obj += alpha * m.demand[t] * (remaining_demand - supply_strat[l])
                obj += beta * m.demand[t] * m.demand[t]
                obj += beta * m.demand[t] * (sum(trader_strat[:, t]) - trader_strat[i, t])
        return obj
    model.obj = Objective(rule=obj_rule, sense=minimize)

    # Solve
    solver = SolverFactory('ipopt')
    result = solver.solve(model, tee=False)

    if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition != TerminationCondition.optimal):
        print("Pyomo failed with status:", result.solver.status)
        return None
    else:
        demand_val = np.array([value(model.demand[t]) for t in model.T])
        return demand_val

def get_seller_best_response(game_dict, trader_strat):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]

    model = ConcreteModel()
    model.T = RangeSet(0, T-1)

    # Variables
    model.supply = Var(model.T, domain=NonNegativeReals)

    # Objective
    def obj_rule(m):
        obj = 0
        for t in m.T:
            obj += p_0 * m.supply[t]
            for l in range(t+1):
                obj += alpha * m.supply[t] * (sum(trader_strat[:, l]) - m.supply[l])
            obj += beta * m.supply[t] * sum(trader_strat[:, t])
        return obj
    model.obj = Objective(rule=obj_rule, sense=maximize)

    # Solve
    solver = SolverFactory('ipopt')
    result = solver.solve(model, tee=False)

    if (result.solver.status != SolverStatus.ok) or (result.solver.termination_condition != TerminationCondition.optimal):
        print("Pyomo failed with status:", result.solver.status)
        return None
    else:
        demand_val = np.array([value(model.supply[t]) for t in model.T])
        return demand_val


def get_cost(game_dict, demand_matrix, supply):
    price_vector = walruss_model(demand_matrix, supply, game_dict)
    total_costs = []
    for i in range(game_dict["n"]):
        cost = np.dot(price_vector, demand_matrix[i])
        total_costs.append(cost) 
    return price_vector, total_costs
   

def equilibrium_test_symmetric():
    n = 2
    game_dict = {
        "n" :   n,
        "T" :   3,
        "p_0" : 0,
        "Vs" : [10 for i in range(n)],
        "alpha" : 1
    }
    supply = [5, 5]
    
    symmetric_opt = get_trader_best_response_scipy(
        game_dict, 
        np.zeros(shape=(game_dict["n"], game_dict["T"])), 
        supply, 
        0, 
        symmetric=True
    ) 
    symmetric_demand = np.array([symmetric_opt for i in range(n)])
    price_vector, total_cost = get_cost(game_dict, symmetric_demand, supply)
    print(f"The symmetric_demand matrix is: {symmetric_demand}")
    print(f"Symmetric strat leads to price: {price_vector} and total cost: {np.sum(total_cost)} to all traders") 

    # Now try and find a Nash Equilibrium through best-response play. The starting symmetric
    # position is just an intial point.
    j, demand_matrix = 0, symmetric_demand
    while True:
        update = False
        for i in range(n):
            br_demand_i = get_trader_best_response_scipy(game_dict, demand_matrix, supply, i)
            if np.linalg.norm(br_demand_i - demand_matrix[i]) >= 0.01:
                demand_matrix[i] = br_demand_i
                update = True
        j += 1
        print(f"Completed round: {j}")
        if not update:
            print("Found Equilibrium")
            break

    price_vector, total_cost = get_cost(game_dict, demand_matrix, supply)
    print(f"The equilibrium demand matrix is: {demand_matrix}")
    print(f"This leads to price {price_vector} giving total costs: {np.sum(total_cost)} to traders")


def find_equilibrium_br():
    n, T, alpha, beta = 2, 3, 1, 0.5
    Vs = [10 for i in range(n)]
    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : 0,
        "Vs" : Vs,
        "alpha" : alpha,
        "beta" : beta
    }

    # Initialize supply
    supply = [5, 5, 5]

    # create an initial demand matrix where everyone buys the whole order up-front.
    demand_matrix = np.zeros((n, T))
    for i in range(n):
        demand_matrix[i, 0] = Vs[i]
    
    # Now try and find a Nash Equilibrium through best-response play.
    iter, eps, supply_player = 0, 0.01, True
    while True:
        update = False
        for i in range(n):
            br_demand_i = get_trader_best_response_pyomo(game_dict, demand_matrix, supply, i)
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

    price_vector, total_cost = get_cost(game_dict, demand_matrix, supply)
    revenue = np.dot(price_vector, supply)

    print(f"The equilibrium supply is: {supply}")
    print(f"The equilibrium demand matrix is: {demand_matrix}\n\n")
    print(f"This leads to price {price_vector}")
    print(f"The cost to each trader is: {total_cost}")
    print(f"The total cost is: {sum(total_cost)}")
    print(f"The revenue is: {revenue}")

def best_response_test():
    n = 2
    game_dict = {
        "n" :   n,
        "T" :   2,
        "beta" : 0,
        "p_0" : 0,
        "Vs" : [10 for i in range(n)],
        "alpha" : 1
    }
    
    supply = [5, 5]
    demand_matrix = np.array([
        [10, 0],
        [10, 0]
    ])
    
    price_vector, total_cost = get_cost(game_dict, demand_matrix, supply)
    print(f"For demand matrix: {demand_matrix}")
    print(f"The prices are: {price_vector} and trader costs are: {total_cost}") 
    
    br_demand_0 = get_trader_best_response_pyomo(game_dict, demand_matrix, supply, 0)
    #br_demand_1 = get_trader_best_response_pyomo(game_dict, demand_matrix, supply, 1) 
    print(f"Trader 0 br: {br_demand_0}")
    #print(f"Trader 1 br: {br_demand_1}")

    br_demand_matrix = np.array([
        br_demand_0, demand_matrix[1]
    ])
    price_vector, total_cost = get_cost(game_dict, br_demand_matrix, supply)
    revenue = np.dot(price_vector, supply)
    print(f"For demand matrix: {br_demand_matrix}")
    print(f"The prices are: {price_vector} and trader costs are: {total_cost}") 
    print(f"The supplier revenue is: {revenue}")

if __name__ == "__main__":
    find_equilibrium_br()
    