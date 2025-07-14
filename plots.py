import numpy as np
import matplotlib.pyplot as plt

from main import find_equilibrium_br

def plot_positions_subplot(ax, game_dict, demand_matrix, supply, cumulative=True, ylabel=False):
    """
    Plot the cumulative positions for all buyers and supplier on a axis.
    
    Args:
        ax: Matplotlib subplot axis
        game_dict: Dictionary containing game parameters
        demand_matrix: Final equilibrium demand matrix (n x T)
        supply: Final equilibrium supply vector (T)
        iteration: Final iteration number
        alpha_val: Current alpha value
        beta_val: Current beta value
        eq_found: Boolean indicating whether equilibrium was found
        max_iter: Maximum number of iterations
        supply_player: Boolean indicating whether supply is a player
    """
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, exp = game_dict["alpha"], game_dict["beta"], game_dict["exp"]
    time_steps = list(range(T))
    
    # Custom color palette
    trader_color = 'blue'
    supplier_color = 'red'
    
    # Create evenly spaced opacities for n traders
    trader_opacities = np.linspace(0.3, 0.9, n)
    
    # Calculate cumulative positions for each buyer
    for i in range(n):
        cumulative_demand = np.cumsum(demand_matrix[i])
        ax.plot(time_steps, cumulative_demand, 
                linewidth=1.5, markersize=4, alpha=trader_opacities[i],
                color=trader_color, 
                label=f'Buyer {i+1}')
    
    # Calculate cumulative supply
    if supply:
        cumulative_supply = np.cumsum(supply)
        ax.plot(time_steps, cumulative_supply, 
                linewidth=1.5, markersize=4,
                color=supplier_color, linestyle='--',
                label='Supplier')
    
    # Customize the subplot
    ax.set_xlabel('Time Steps', fontsize=10)
   
    if cumulative:
        if ylabel:
            ax.set_ylabel('Cum Position', fontsize=10)
        title = f'exp:{exp} Cumulative Position'
    else:
        if ylabel:
            ax.set_ylabel('Order', fontsize=10)
        title = f'α={alpha}, β={beta}, exp: {exp} Orders'

    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(-1, 18)
    ax.set_yticks(np.linspace(0, 16, 9))
    ax.set_xticks(np.linspace(0, T-1, T))

def run_sweep_alpha_beta(game_dict, alpha_range, beta_range, supply_player=True):
    n, T = game_dict["n"], game_dict["T"]
    p_0 = game_dict["p_0"]
    Vs = game_dict["Vs"]

    fig1, axes1 = plt.subplots(len(beta_range), len(alpha_range), 
                              figsize=(15, 8), sharex=True, sharey=True)
    for i, beta in enumerate(beta_range):
        for j, alpha in enumerate(alpha_range):
            game_dict["alpha"] = alpha
            game_dict["beta"] = beta
            found, demand_matrix_eq, supply_eq = find_equilibrium_br(game_dict, supply_player)
            if not supply_player:
                supply_eq = None
            if found:
                if len(beta_range) == 1:
                    plot_positions_subplot(axes1[j], game_dict, demand_matrix_eq, supply_eq, ylabel=True if j == 0 else False)
                else:
                    plot_positions_subplot(axes1[i, j], game_dict, demand_matrix_eq, supply_eq, ylabel=True if j == 0 else False)
    
    fig1.suptitle(f'n={n}, T={T}, Vs={Vs}, p_0={p_0}', fontsize=16)
    plt.tight_layout()
    plt.show()


def run_sweep_exp(game_dict, exp_range):
    n, T = game_dict["n"], game_dict["T"]
    p_0 = game_dict["p_0"]
    Vs = game_dict["Vs"]
    supply = game_dict["supply"]

    fig1, axes1 = plt.subplots(1, len(exp_range), figsize=(22, 8), sharex=True, sharey=True)
    for i, exp in enumerate(exp_range):
        game_dict["exp"] = exp
        found, demand_matrix_eq, _ = find_equilibrium_br(game_dict, supply, get_welfare=False)
        plot_positions_subplot(axes1[i], game_dict, demand_matrix_eq, supply, ylabel=True if i == 0 else False) 
    
    fig1.suptitle(f'n={n}, T={T}, Vs={Vs}, p_0={p_0}, exp:{exp_range}', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n, T, alpha, beta = 2, 5, 3, 1
    Vs = [12, 15]
    supply = [0 for i in range(T)]
    game_dict = {
        "n" :   n,
        "T" :   T,
        "p_0" : 5,
        "Vs" : Vs,
        "alpha" : alpha/T,
        "beta" : beta,
        "reserve" : None,
        "exp" : 1,
        "supply" : supply
    }
    exp_range = [0.3, 0.5, 0.75, 0.95, 1]
    
    run_sweep_exp(game_dict, exp_range)
