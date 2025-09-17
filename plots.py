import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from main import find_equilibrium_br, get_cost
from algorithms import extra_gradient_equilibrium, extra_gradient_equilibrium_bayesian

import imageio.v2 as imageio
import os, re, glob
from tqdm import tqdm


# Enable LaTeX text rendering globally
plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\boldmath\bfseries' # or other packages that support bold

# Set the font family (e.g., to serif fonts often used with LaTeX)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or other serif fonts

plt.rcParams.update({'font.size': 13}) # Default font size for most text

# # Specific font sizes for different text elements
# plt.rcParams['axes.titlesize'] = 14     # Font size of the axes title
# plt.rcParams['axes.labelsize'] = 12     # Font size of the x and y labels
# plt.rcParams['xtick.labelsize'] = 10    # Font size of the x-axis tick labels
# plt.rcParams['ytick.labelsize'] = 10    # Font size of the y-axis tick labels
# plt.rcParams['legend.fontsize'] = 10    # Font size of the legend
# plt.rcParams['figure.titlesize'] = 16   # Font size of the figure title


def plot_price(ax, game_dict, demand_matrix, supply, plot_y_label=True):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    
    # Get price vector
    price_vector, perm_price_vector, total_cost = get_cost(game_dict, demand_matrix)
    time_steps = np.arange(T)

    # Color palette for players
    colors = ['red', 'pink']
    
    price_delta = price_vector
    perm_price_delta = perm_price_vector
    ax.plot(time_steps, price_delta, linewidth=2, color=colors[0], label=r'\textbf{Exec Price: $p_t$}')
    ax.plot(time_steps, perm_price_delta, linewidth=2, color=colors[1], label=r'\textbf{Perm Price: $p_t^w$}')
    ax.set_xlabel(r'\textbf{Time}')
    if plot_y_label:
        ax.set_ylabel(r'\textbf{Price}')
    ax.grid(True, alpha=0.25)
    ax.legend()

def plot_demand(ax, game_dict, demand_matrix, supply, plot_type, plot_y_label=True):
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    
    # Get price vector
    price_vector, perm_price_vector, total_cost = get_cost(game_dict, demand_matrix)
    time_steps = np.arange(T)

    # Color palette for players
    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray']
    to_plot_demand = demand_matrix
    to_plot_supply = supply

    if plot_type == "cumulative":
        to_plot_demand = np.cumsum(demand_matrix, axis=1)
        to_plot_supply = np.cumsum(supply) 
   
    for i in range(n):
        ax.plot(time_steps, to_plot_demand[i], linewidth=2,
            color=colors[i % len(colors)],
            label=(r'\textbf{Trader} ' + f'{i}' + r' \textbf{(V=}' + rf'{Vs[i]}' + r'\textbf{)}')
        )
    ax.plot(time_steps, to_plot_supply, linewidth=2, linestyle=":", color=colors[-1], label=r"\textbf{Noise Agent}")
    ax.set_xlabel(r'\textbf{Time}')
    if plot_type == "cumulative" and plot_y_label:
        ax.set_ylabel(r'\textbf{Cumulative Position}') 
    elif plot_y_label:
         ax.set_ylabel('Order')
    ax.set_title(rf'$\alpha={alpha}$, $\beta={beta}$')
    ax.set_ylim(-15, 35) # Set y-axis limit
    ax.legend()
    ax.grid(True, alpha=0.25)  


def plot_demand_bayesian(ax, game_dict, demand_matrix, supply, plot_type, plot_y_label=True):
    n, k, T = game_dict["n"], game_dict["k"], game_dict["T"]
    alphas, betas, p_0 = game_dict["alphas"], game_dict["betas"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserves = game_dict["reserves"]
    
    time_steps = np.arange(T)

    colors = ['blue', 'orange']
    to_plot_demand = demand_matrix
    to_plot_supply = supply

    if plot_type == "cumulative":
        to_plot_demand = np.cumsum(demand_matrix, axis=2)
        to_plot_supply = np.cumsum(supply) 
   
    for i in range(n):
        for l in range(k):
            v, r = Vs[i,l], reserves[i,l]
            ax.plot(time_steps, to_plot_demand[i][l], linewidth=2,
                color=colors[i % len(colors)], alpha=(l*0.3 + 0.4),
                label=(r'\textbf{Agent} ' + f'{i}' + rf' $\theta_{i}=({v},{r:.1f})$')
            )
    ax.set_xlabel(r'\textbf{Time}')
    if plot_type == "cumulative" and plot_y_label:
        ax.set_ylabel(r'\textbf{Cumulative Position}') 
    elif plot_y_label:
         ax.set_ylabel('Order')
    
    alpha = np.min(alphas)
    beta_min = np.min(betas)
    beta_max = np.max(betas)
    ax.set_title(rf'$\alpha={alpha}$, $\beta \in [{beta_min}, {beta_max}]$')
    ax.set_ylim(-15, 35) # Set y-axis limit
    ax.legend()
    ax.grid(True, alpha=0.25)  


def plot_equilibrium_strategies(game_dict, demand_matrix, supply, exp_number=None, beta_ab=None, time_ab=None):
    """
    Plot the equilibrium strategies showing:
    1. Individual player strategies over time
    2. Cumulative positions for each player
    3. Supply over time
    4. Price evolution over time (with reserve prices)
    """
    discretization, cont_time_interval = game_dict["discretization"], game_dict["cont_time_interval"]
    n, T = game_dict["n"], game_dict["T"]
    alpha, beta, p_0 = game_dict["alpha"], game_dict["beta"], game_dict["p_0"]
    Vs = game_dict["Vs"]
    reserve = game_dict["reserve"]
    
    # Get price vector
    price_vector, perm_price_vector, total_cost = get_cost(game_dict, demand_matrix)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    time_steps = np.arange(T)
    
    # Color palette for players
    colors = ['blue', 'orange', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    ticks = np.arange(0, (cont_time_interval+1) * discretization, discretization)
    
    # 1. Individual player strategies (orders per time step)
    ax1 = axes[0, 0]
    plot_demand(ax1, game_dict, demand_matrix, supply, "order")
    
    # 2. Cumulative positions (with alternating plotting order)
    ax2 = axes[0, 1]
    plot_demand(ax2, game_dict, demand_matrix, supply, "cumulative")

    # Plot costs
    ax3 = axes[1, 0]
    # Elementwise multiply each row of demand_matrix by price_vector
    costs_per_step = demand_matrix * price_vector  # shape (n, T)

    # Cumulative sum along time axis (axis=1)
    cost_matrix = np.cumsum(costs_per_step, axis=1)  # shape (n, T)
    for i in range(n):
        ax3.plot(time_steps, cost_matrix[i], linewidth=2,
            color=colors[i % len(colors)],
            label=(f'Trader {i} (V={Vs[i]}, R={reserve[i]})')
        )
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Cumulative cost upto t')
    ax3.set_title('Cumulative Costs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels((ticks // discretization).astype(int))

    # 4. Price evolution (with reserve prices)
    ax4 = axes[1, 1]
    plot_price(ax4, game_dict, demand_matrix, supply)
    
    # Add overall title
    fig.suptitle(f'Equilibrium Analysis: n:{n}, Discretization steps: {discretization}, Cont time: {cont_time_interval}, T: {discretization}*{cont_time_interval}, α={alpha}, β={beta/discretization:.3f}*{discretization})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if exp_number:
        if time_ab:
            plt.savefig(f"figures/exp{exp_number}_T/exp{exp_number}_T_{cont_time_interval}.png")
        elif beta_ab:
            beta_int = int(beta/discretization * 1000)
            plt.savefig(f"figures/exp{exp_number}_beta/exp{exp_number}_beta_{beta_int}.png")
        plt.close()
    else:
        plt.show()


def extract_number(filename):
    # This regex finds the last group of digits in the filename
    match = re.search(r'(\d+)(?=\.png$)', filename)
    return int(match.group(1)) if match else -1


def bayesian_experiment():
    n, k, T = 2, 3, 100
    supply = [0 for i in range(T)]
    bayesian_game_dict = {
        "n" : n,
        "T" : T,
        "k" : k,
        "p_0" : 2,
        "supply" : supply
    }

    # Vs and reserves are an n (agent) x k (type) matrix. 
    Vs = np.array([
        [10, 15, 20],
        [20, 25, 30]
    ])
    reserves = Vs/3
    bayesian_game_dict["Vs"] = Vs
    bayesian_game_dict["reserves"] = reserves

    # key is agent1 type, agent2 type
    # All that really matters is the expected value of alpha, beta conditioned on the type. Which is what this is
    alphas, betas, type_dist = np.zeros((k,k)), np.zeros((k,k)), np.zeros((k,k))
    for l0 in range(k):
        for l1 in range(k):
            key = (l0, l1)
            beta = 0.5*(bayesian_game_dict["Vs"][(0,l0)] + bayesian_game_dict["Vs"][(1,l1)])/200
            alpha = 0.1
            alphas[l0, l1] = alpha
            betas[l0, l1] = beta
            type_dist[l0, l1] = 1/k**2

    bayesian_game_dict["alphas"] = alphas
    bayesian_game_dict["type_dist"] = type_dist

    beta_multiplier = [1, 10, 100]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, mult in enumerate(beta_multiplier):
        bayesian_game_dict["betas"] = mult*betas
        demand_matrix = extra_gradient_equilibrium_bayesian(bayesian_game_dict)
        plot_demand_bayesian(axes[i], bayesian_game_dict, demand_matrix, supply, "cumulative", plot_y_label=(True if i==0 else False))
    plt.tight_layout()
    plt.show()


def complete_information_experiment():
    # If you're doing cont time:
    # choose discretization = d
    # choose cont_time_interval = c
    # choose T = d*c
    # choose beta*discretization
    n, alpha = 5, 0.1
    T = 100
    Vs = [10, 15, 20, 25, 30]
    reserve = [4, 5, 6, 7, 8]
    beta_range = [0.1, 1, 10]
    supply = np.random.randn(T)*0.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, beta in enumerate(beta_range):
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
        demand_matrix = extra_gradient_equilibrium(game_dict)
        plot_demand(axes[i], game_dict, demand_matrix, supply, "cumulative", plot_y_label=(True if i==0 else False))
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, beta in enumerate(beta_range):
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
        demand_matrix = extra_gradient_equilibrium(game_dict)
        plot_price(axes[i], game_dict, demand_matrix, supply, plot_y_label=(True if i==0 else False))
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    #complete_information_experiment()
    bayesian_experiment()

    # print("Completed generating plots") 
    # image_files = sorted(glob.glob(f'figures/exp{exp_number}_{ablation}/exp{exp_number}_{ablation}_*.png'), key=extract_number)
    # with imageio.get_writer(f'ablation_exp{exp_number}_{ablation}.gif', mode='I', fps=1.8) as writer:
    #     for filename in image_files:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

