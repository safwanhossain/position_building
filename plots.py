import numpy as np
import matplotlib.pyplot as plt

from main import find_equilibrium_br, get_cost
from algorithms import extra_gradient_equilibrium

import imageio.v2 as imageio
import os, re, glob
from tqdm import tqdm


def plot_positions_subplot(ax, game_dict, demand_matrix, cumulative=True, ylabel=False):
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
    supply = game_dict["supply"]
    time_steps = list(range(T))
    
    # Custom color palette
    trader_color = 'blue'
    supplier_color = 'red'
    
    # Create evenly spaced opacities for n traders
    trader_opacities = np.linspace(0.3, 0.9, n)
    
    # Calculate cumulative positions for each buyer
    for i in range(n):
        if cumulative:
            cumulative_demand = np.cumsum(demand_matrix[i])
            ax.plot(time_steps, cumulative_demand, 
                    linewidth=1.5, markersize=4, alpha=trader_opacities[i],
                    color=trader_color, 
                    label=f'Buyer {i+1}')
        else:
            ax.plot(time_steps, demand_matrix[i], 
                    linewidth=1.5, markersize=4, alpha=trader_opacities[i],
                    color='tab:red', 
                    label=f'Buyer {i+1}')
       
    
    # Calculate cumulative supply
    if np.sum(supply) != 0:
        cumulative_supply = np.cumsum(supply)
        ax.plot(time_steps, cumulative_supply, 
                linewidth=1.5, markersize=4,
                color=supplier_color, linestyle='--',
                label='Supplier')
    
    # Customize the subplot
    ax.set_xlabel('Time Steps', fontsize=10)
   
    if cumulative:
        if ylabel:
            ax.set_ylabel('Cumulative Position', fontsize=10)
    else:
        if ylabel:
            ax.set_ylabel('Order', fontsize=10)
    title = f'β={beta}, exp: {exp}'

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


def run_sweep_exp(game_dict, beta_range, exp_range):
    n, T = game_dict["n"], game_dict["T"]
    p_0 = game_dict["p_0"]
    Vs = game_dict["Vs"]

    #fig1, axes1 = plt.subplots(len(beta_range), len(exp_range), figsize=(22, 8), sharex=True, sharey=True)
    fig1, axes1 = plt.subplots(1, len(beta_range), figsize=(22, 8), sharex=True, sharey=True)
    for i, beta in enumerate(beta_range):
        for j, exp in enumerate(exp_range):
            game_dict["exp"] = exp
            game_dict["beta"] = beta
            found, demand_matrix_eq, _ = find_equilibrium_br(game_dict, get_welfare=False)
            plot_positions_subplot(axes1[i], game_dict, demand_matrix_eq, cumulative=True, ylabel=True) 
    
    fig1.suptitle(f'n={n}, T={T}, Vs={Vs}, p_0={p_0}, exp:{exp_range}, beta:{beta_range}', fontsize=16)
    plt.tight_layout()
    plt.show()

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
    
    # 1. Individual player strategies (orders per time step)
    ax1 = axes[0, 0]
    for i in range(n):
        ax1.plot(time_steps, demand_matrix[i], linewidth=2,
            color=colors[i % len(colors)],
            label=(f'Player {i} (V={Vs[i]})')
        )
    ax1.plot(time_steps, supply, linewidth=2, linestyle=":", color=colors[-1], label="Supply")
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Order Size')
    ax1.set_title('Player Orders at Equi')
    ax1.legend()
    ticks = np.arange(0, (cont_time_interval+1) * discretization, discretization)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels((ticks // discretization).astype(int))
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative positions (with alternating plotting order)
    ax2 = axes[0, 1]
    cumulative_demand_matrix = np.cumsum(demand_matrix, axis=1)
    cumulative_supply = np.cumsum(supply)
    for i in range(n):
        ax2.plot(time_steps, cumulative_demand_matrix[i], linewidth=2,
            color=colors[i % len(colors)],
            label=(f'Player {i} (V={Vs[i]})')
        )
    ax2.plot(time_steps, cumulative_supply, linewidth=2, linestyle=":", color=colors[-1], label="Cumulative Supply")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Position')
    ax2.set_title('Player Cumulative Pos at Equi')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((ticks // discretization).astype(int))

    # Plot costs
    ax3 = axes[1, 0]
    # Elementwise multiply each row of demand_matrix by price_vector
    costs_per_step = demand_matrix * price_vector  # shape (n, T)

    # Cumulative sum along time axis (axis=1)
    cost_matrix = np.cumsum(costs_per_step, axis=1)  # shape (n, T)
    for i in range(n):
        ax3.plot(time_steps, cost_matrix[i], linewidth=2,
            color=colors[i % len(colors)],
            label=(f'Player {i} (V={Vs[i]}, R={reserve[i]})')
        )
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Cumulative cost upto t')
    ax3.set_title('Cumulative Costs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels((ticks // discretization).astype(int))

    # 4. Price evolution (with reserve prices)
    price_delta = price_vector
    perm_price_delta = perm_price_vector
    ax4 = axes[1, 1]
    ax4.plot(time_steps, price_delta, linewidth=2, color=colors[-2], label=r'Exec Price: $p_t$')
    ax4.plot(time_steps, perm_price_delta, linewidth=2, color=colors[-3], label=r'Perm Price: $p_t^w$')
  
    # Plot each player's reserve price if available
    # if reserve is not None:
    #     for i in range(n):
    #         ax4.axhline(y=reserve[i], color=colors[i + 2], linestyle=':', alpha=0.8,
    #                     label=f'Reserve {i} ({reserve[i]}) - p_0')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.set_title(rf'Price Evolution; Reserve {reserve}; $p_0=${p_0}')
    ax4.legend()
    ax4.set_xticks(ticks)
    ax4.set_xticklabels((ticks // discretization).astype(int))
    ax4.grid(True, alpha=0.3)
    
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


if __name__ == "__main__":
    exp_number = 1
    ablation = "T"
    os.makedirs(f'figures/exp{exp_number}_{ablation}', exist_ok=True)
    
    if ablation == "T":
        for cont_time_interval in tqdm(np.arange(2, 21, 1)):
            discretization = 100
            cont_time_interval = 1
            n, alpha, beta = 3, 1.0, 0.1
            T = discretization*cont_time_interval
            #Vs = [75 for i in range(15)] + [10 for i in range(5)]
            #reserve = [500 for i in range(n)]
            Vs = [10, 40, 80]
            reserve = [5000, 5000, 5000]
            #supply = [0 for _ in range(T)]
            
            total_supply_cap = sum(Vs)*0.8

            # Generate random numbers
            random_supply = np.random.rand(T)

            # Scale so that sum is less than total_supply_cap (e.g., 80% of cap)
            scale = 0.8 * total_supply_cap / np.sum(random_supply)
            random_supply = random_supply * scale
            supply = random_supply
            # # Generate a sinusoidal pattern (values between 0 and 1)
            # x = np.linspace(0, 8 * np.pi, T)
            # sinusoid = (np.sin(x) + 1) / 2  # Shift to range [0, 1]

            # # Scale so that sum is less than total_supply_cap (e.g., 80% of cap)
            # scale = 0.8 * total_supply_cap / np.sum(sinusoid)
            # supply = sinusoid * scale

            game_dict = {
                "discretization" : discretization,
                "cont_time_interval" : cont_time_interval,
                "n" :   n,
                "T" :   T,
                "p_0" : 2.0,
                "Vs" : Vs,
                "alpha" : alpha,
                "beta" : beta*discretization,
                "supply" : supply,
                "reserve" : reserve,
                "exp" : 1
            }
            demand_matrix = extra_gradient_equilibrium(game_dict)
            #plot_equilibrium_strategies(game_dict, demand_matrix, supply, exp_number=exp_number, time_ab=cont_time_interval)
            plot_equilibrium_strategies(game_dict, demand_matrix, supply)

    else:
        for beta in tqdm(np.arange(0.1, 0.525, 0.025)):
            discretization = 50
            cont_time_interval = 10
            n, alpha = 5, 0.1
            T = discretization*cont_time_interval
            Vs = [10, 15, 20, 25, 30]
            reserve = [3, 3.5, 4, 4.5, 5]
            supply = np.array([x * sum(Vs) / sum(np.exp(np.linspace(0, -1, T))) for x in np.exp(np.linspace(0, 3, T))])*0.02

            game_dict = {
            "discretization" : discretization,
            "cont_time_interval" : cont_time_interval,
            "n" :   n,
            "T" :   T,
            "p_0" : 2.0,
            "Vs" : Vs,
            "alpha" : alpha,
            "beta" : beta*discretization,
            "supply" : supply,
            "reserve" : reserve,
            "exp" : 1
            }
            demand_matrix = extra_gradient_equilibrium(game_dict)
            plot_equilibrium_strategies(game_dict, demand_matrix, supply, exp_number=exp_number, beta_ab=beta)


    print("Completed generating plots") 
    image_files = sorted(glob.glob(f'figures/exp{exp_number}_{ablation}/exp{exp_number}_{ablation}_*.png'), key=extract_number)
    with imageio.get_writer(f'ablation_exp{exp_number}_{ablation}.gif', mode='I', fps=1.8) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

