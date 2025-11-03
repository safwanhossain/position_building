"""
Multi-Agent Optimal Position Building Environment

This module implements a Markov Game environment for strategic optimal execution

The environment supports:
- Multiple agents with independent policies
- Dynamic market parameters (α, β, exogenous supply)
- Configurable utility functions and transition distributions
- Both finite and infinite horizon settings
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from copy import deepcopy

@dataclass
class State:
    """
    State representation for the Markov Game.
    
    Attributes:
        positions: Current cumulative position h_i for each agent i
        price: Current market price p
        exogenous_supply: Current exogenous and instanteneous supply s
        alpha: Current price impact parameter (permanent impact)
        beta: Current price impact parameter (temporary impact)
    """  
    positions: np.ndarray  # shape: (n_agents,)
    price: float
    exogenous_supply: float
    alpha: float
    beta: float
    time: int
    horizon: int
    
    def copy(self) -> 'State':
        """Create a deep copy of the state."""
        return State(
            positions=self.positions.copy(),
            price=self.price,
            exogenous_supply=self.exogenous_supply,
            alpha=self.alpha,
            beta=self.beta,
            time=self.time,
            horizon=self.horizon
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            'positions': self.positions.copy(),
            'price': self.price,
            'exogenous_supply': self.exogenous_supply,
            'alpha': self.alpha,
            'beta': self.beta,
            'time': self.time,
            'horizon': self.horizon
        }


class TransitionDistribution(ABC):
    """
    Abstract base class for transition distributions P(α', β', s' | ω).
    This allows for easy swapping of different market dynamics models.
    """
    
    @abstractmethod
    def sample(self, state: State) -> Tuple[float, float, float]:
        """
        Sample next values for (α', β', s') given current state.
        
        Args:
            state: Current state ω
            rng: Random number generator for reproducibility
            
        Returns:
            Tuple of (alpha', beta', s')
        """
        pass
    
    @abstractmethod
    def get_mean(self, state: State) -> Tuple[float, float, float]:
        """Get mean/expected values for (α', β', s') given current state."""
        pass


class ConstantTransition(TransitionDistribution):
    """Transition distribution where α, β remain constant.
       The exogenous supply variable is sampled from a gaussian distribution with 0 mean.
       Notice that this is state independent.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def sample(self, state: State) -> Tuple[float, float, float]:
        supply = np.random.normal(loc=self.mean, scale=self.std, size=1)
        return state.alpha, state.beta, supply
    
    def get_mean(self, state: State) -> Tuple[float, float, float]:
        return state.alpha, state.beta, self.mean

class IndependentGaussianTransition(TransitionDistribution):
    """Transition distribution where α, β, s are all sampled from an independent gaussian distribution.
       Notice that this is state independent.
    """
    def __init__(self, alpha_mean, alpha_std, beta_mean, beta_std, s_mean, s_std):
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.beta_mean = beta_mean
        self.beta_std = beta_std
        self.s_mean = s_mean
        self.s_std = s_std
    
    def sample(self, state: State) -> Tuple[float, float, float]:
        alpha = np.random.normal(loc=self.alpha_mean, scale=self.alpha_std, size=1)
        beta = np.random.normal(loc=self.beta_mean, scale=self.beta_std, size=1)
        supply = np.random.normal(loc=self.s_mean, scale=self.s_std, size=1)
        return alpha, beta, supply
    
    def get_mean(self, state: State) -> Tuple[float, float, float]:
        return self.alpha_mean, self.beta_mean, self.s_mean


class ARGaussianTransition(TransitionDistribution):
    """
    Gaussian transition with AR(1) dynamics for each parameter.
    The AR(1) model assumes that the current value is a linear combination of the past value
    plus some noise term. 

    So X_{t+1} = a_0 + a_1*X_t + ε. Here X can be alpha, beta or supply.
    where ε ~ N(0, σ²)

    TODO: Implement this properly
    """
    
    def __init__(
        self,
        mean_alpha: float = 0.1,
        mean_beta: float = 0.01,
        mean_supply: float = 0.0,
        persistence_alpha: float = 0.9,
        persistence_beta: float = 0.9,
        persistence_supply: float = 0.8,
        std_alpha: float = 0.01,
        std_beta: float = 0.001,
        std_supply: float = 1.0
    ):
        self.mean_alpha = mean_alpha
        self.mean_beta = mean_beta
        self.mean_supply = mean_supply
        self.persistence_alpha = persistence_alpha
        self.persistence_beta = persistence_beta
        self.persistence_supply = persistence_supply
        self.std_alpha = std_alpha
        self.std_beta = std_beta
        self.std_supply = std_supply
    
    def sample(self, state: State):
        pass

    def get_mean(self, state: State) -> Tuple[float, float, float]:
        pass

    # def sample(self, state: State, rng: np.random.Generator) -> Tuple[float, float, float]:
    #     alpha_next = (
    #         self.mean_alpha + 
    #         self.persistence_alpha * (state.alpha - self.mean_alpha) +
    #         rng.normal(0, self.std_alpha)
    #     )
    #     beta_next = (
    #         self.mean_beta + 
    #         self.persistence_beta * (state.beta - self.mean_beta) +
    #         rng.normal(0, self.std_beta)
    #     )
    #     supply_next = (
    #         self.mean_supply + 
    #         self.persistence_supply * (state.exogenous_supply - self.mean_supply) +
    #         rng.normal(0, self.std_supply)
    #     )
        
    #     # Ensure α, β are positive
    #     alpha_next = max(alpha_next, 1e-6)
    #     beta_next = max(beta_next, 1e-6)
        
    #     return alpha_next, beta_next, supply_next
    
    # def get_mean(self, state: State) -> Tuple[float, float, float]:
    #     alpha_next = self.mean_alpha + self.persistence_alpha * (state.alpha - self.mean_alpha)
    #     beta_next = self.mean_beta + self.persistence_beta * (state.beta - self.mean_beta)
    #     supply_next = self.mean_supply + self.persistence_supply * (state.exogenous_supply - self.mean_supply)
    #     return alpha_next, beta_next, supply_next


class Constraint(ABC):
    """Abstract base class for feasibility constraints G_i."""
    
    @abstractmethod
    def is_feasible(self, cum_position: float) -> bool:
        """Check if position satisfies the constraint."""
        pass
    
    @abstractmethod
    def project(self, cum_position: float) -> float:
        """Project position onto feasible set."""
        pass


class BoxConstraint(Constraint):
    """Box constraint: h_min ≤ h ≤ h_max"""
    
    def __init__(self, min_pos: float = -np.inf, max_pos: float = np.inf):
        self.h_min = min_pos
        self.h_max = max_pos
    
    def is_feasible(self, cum_position: float) -> bool:
        return self.h_min <= cum_position <= self.h_max
    
    def project(self, cum_position: float) -> float:
        return np.clip(cum_position, self.h_min, self.h_max)
    

class UtilityFunction(ABC):
    """
    Abstract base class for agent utility functions f_i(h_i).
    Each agent has an idiosyncratic utility for holding position h_i.
    """
    
    @abstractmethod
    def __call__(self, cum_position: float) -> float:
        """Evaluate utility at given cumulative position."""
        pass
    
    @abstractmethod
    def gradient(self, cum_position: float) -> float:
        """Compute gradient of utility function."""
        pass

class ReserveUtility(UtilityFunction):
    """Simple linear utility: f(h) = reserve_price * h"""
    
    def __init__(self, reserve_price: float):
        self.reserve_price = reserve_price
    
    def __call__(self, cum_position: float) -> float:
        return self.reserve_price * cum_position
    
    def gradient(self, cum_position: float) -> float:
        return self.reserve_price

class Policy(ABC):
    """
    Abstract base class for agent policy functions \pi(a_i | \omega).
    """
    @abstractmethod
    def __call__(self, curr_state: State) -> float:
        """Return the sampled action from that state"""
        pass

class UniformPolicy(Policy):
    def __init__(self, utility_func, constraint, agent_id):
        self.utility_func = utility_func
        self.constraint = constraint
        self.agent_id = agent_id
        pass

    def __call__(self, curr_state: State) -> float:
        # Buy a uniform amount if the price is below reserve.
        # Don't buy anything if the price is above reserve
        if curr_state.price > self.utility_func.gradient(curr_state.positions[self.agent_id]):
            return 0
    
        horizon = curr_state.horizon
        return self.constraint.h_max / curr_state.horizon

class OptimalExecutionEnv:
    """
    Multi-Agent Markov Game environment for optimal position building.
    
    This implements the environment described in the paper,
    supporting both finite and infinite horizon settings.
    """
    
    def __init__(
        self,
        n_agents: int,
        utility_functions: List[UtilityFunction],
        feasibility_constraints: List[Constraint],
        transition_distribution: TransitionDistribution,
        initial_state: State,
        max_episodes: int,
        horizon: int,
        policies: List[Policy],
        discount_factor: float = 1.0,
    ):
        """
        Initialize the environment.
        
        Args:
            n_agents: Number of agents
            utility_functions: List of utility functions f_i for each agent
            feasibility_constraints: List of feasibility constraints G_i for each agent
            transition_distribution: Distribution P(α', β', s' | ω)
            initial_state: Initial state distribution μ (or single initial state)
            horizon: Episode horizon T (None for infinite horizon)
            discount_factor: Discount factor γ
            action_bounds: Tuple of (min_action, max_action) for each agent
            seed: Random seed for reproducibility
        """
        self.n_agents = n_agents
        self.utility_functions = utility_functions
        self.feasibility_constraints = feasibility_constraints
        self.transition_distribution = transition_distribution
        self.initial_state = initial_state.copy()
        self.max_episodes = max_episodes
        self.horizon = horizon
        self.policies = policies
        self.discount_factor = discount_factor
        
        # Validation
        assert len(utility_functions) == n_agents
        assert len(feasibility_constraints) == n_agents
        
        # State tracking
        self.current_state = initial_state
        
        # Episode statistics
        self.episode_rewards = np.zeros((max_episodes, self.n_agents))
        self.curr_episode = 0
    
    def reset(self, seed: Optional[int] = None) -> Tuple[State, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional seed for this episode
            
        Returns:
            initial_state: The initial state
            info: Additional information dictionary
        """        
        # Initialize state (could sample from distribution if needed)
        self.current_state = self.initial_state.copy()
        self.current_state.time = 0
        
        info = {
            'initial_positions': self.current_state.positions.copy(),
            'initial_price': self.current_state.price
        }
        
        return self.current_state, info
    
    def _get_next_state(self, state: State, actions: np.ndarray) -> State:
        """
        Compute next state according to transition kernel in Equation (1).
        
        P(ω'|ω, a) defined by:
        - h'_i = h_i + a_i for all i
        - p' = p + α(s + Σ_i a_i)
        - (α', β', s') ~ P(α', β', s'|ω)
        """
        # Update positions
        next_positions = state.positions + actions
        
        # Update price (permanent impact)
        total_order_flow = state.exogenous_supply + np.sum(actions)
        next_price = state.price + state.alpha * total_order_flow
        
        # Sample next market parameters
        next_alpha, next_beta, next_supply = self.transition_distribution.sample(state)
        
        # Create next state
        next_state = State(
            positions=next_positions,
            price=next_price,
            exogenous_supply=next_supply,
            alpha=next_alpha,
            beta=next_beta,
            time=state.time + 1,
            horizon=self.horizon
        )
        return next_state

    def _get_rewards(self, state: State, actions: np.ndarray) -> np.ndarray:
        """
        Compute rewards for each agent according to Definition 4.
        
        r_i(a_i, ω) = [f_i(h_i + a_i) - f_i(h_i)] - a_i[p + α(s + Σ_i a_i) + β(s + Σ_i a_i)]
        """
        rewards = np.zeros(self.n_agents)
        total_order_flow = state.exogenous_supply + np.sum(actions)
        
        for i in range(self.n_agents):
            # If the action leads to not feasible state, penalize them
            if not self.feasibility_constraints[i].is_feasible(state.positions[i] + actions[i]):
                rewards[i] = -np.inf
                continue
            
            # Utility gain from changing position
            utility_gain = (
                self.utility_functions[i](state.positions[i] + actions[i]) - 
                self.utility_functions[i](state.positions[i])
            )
            
            # Execution cost (permanent + temporary impact)
            execution_cost = actions[i] * (
                state.price + 
                state.alpha * total_order_flow + 
                state.beta * total_order_flow
            )
            rewards[i] = utility_gain - execution_cost
        return rewards

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[State, np.ndarray, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            actions: Array of actions [a_1, ..., a_n] for each agent
            
        Returns:
            next_state: The next state ω'
            rewards: Array of rewards [r_1, ..., r_n] for each agent  
            terminated: Whether episode ended naturally
            info: Additional information dictionary
        """
        assert self.current_state is not None, "Must call reset() before step()"
        assert len(actions) == self.n_agents
        
        # Compute rewards for current step
        rewards = self._get_rewards(self.current_state, actions)
        
        # Compute next state
        next_state = self._get_next_state(self.current_state, actions)
        
        # Check termination conditions. Move on to the next episode when this terminates
        terminated = False
        if self.horizon is not None:
            if next_state.time == self.horizon:
                terminated = True
        
        # Update state
        self.current_state = next_state
        self.episode_rewards[self.curr_episode] += rewards
        
        info = {
            'episode' : self.curr_episode,
            'time' : next_state.time,
            'positions': next_state.positions.copy(),
            'price': next_state.price,
            'alpha': next_state.alpha,
            'beta': next_state.beta,
            'exogenous_supply': next_state.exogenous_supply,
            'total_volume': np.sum(actions),
        }
        return next_state, rewards, terminated, info
        

    def get_state(self) -> State:
        """Get current state."""
        assert self.current_state is not None
        return self.current_state.copy()
    

    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
        """
        if self.current_state is None:
            return None
        
        output = [
            f"\n{'='*60}",
            f"Time: {self.current_state.time}",
            f"Price: {self.current_state.price:.4f}",
            f"Alpha: {self.current_state.alpha:.6f}, Beta: {self.current_state.beta:.6f}",
            f"Exogenous Supply: {self.current_state.exogenous_supply:.4f}",
            f"Positions: {self.current_state.positions}",
            f"Current Cumulative Rewards: {self.episode_rewards[self.curr_episode]}"
            f"{'='*60}\n"
        ]
        output_str = '\n'.join(output)
        
        if mode == 'human':
            print(output_str)
            return None
        else:
            return output_str
    
    def compute_value_function(
        self, 
        policy: Callable[[State, int], np.ndarray],
        num_episodes: int = 1000
    ) -> Dict[str, Any]:
        """
        Placeholder for value function computation.
        
        Estimate V_i(ω) by Monte Carlo sampling under given policy.
        
        Args:
            policy: Policy function that maps (state, agent_id) -> action
            num_episodes: Number of episodes to simulate
            
        Returns:
            Dictionary with value estimates and statistics
        """
        # TODO: Implement Monte Carlo estimation
        raise NotImplementedError("Value function computation to be implemented")
    
    def compute_q_function(
        self,
        policy: Callable[[State, int], np.ndarray],
        num_episodes: int = 1000
    ) -> Dict[str, Any]:
        """
        Placeholder for Q-function computation.
        
        Estimate Q_i(ω, a) by Monte Carlo sampling.
        
        Args:
            policy: Policy function that maps (state, agent_id) -> action
            num_episodes: Number of episodes to simulate
            
        Returns:
            Dictionary with Q-value estimates and statistics
        """
        # TODO: Implement Q-function estimation
        raise NotImplementedError("Q-function computation to be implemented")
    

    def run(self):
        self.curr_episode = 0
        for episode in range(self.max_episodes):
            self.reset()
            for time in range(self.horizon):
                actions = np.zeros(self.n_agents)
                for i in range(self.n_agents):
                    actions[i] = self.policies[i](self.current_state)
                prev_state = self.current_state
                next_state, rewards, terminated, info = self.step(actions)

            assert terminated is True
            print(f"Episode {episode} had agents achieve cumulative reward: {self.episode_rewards[episode]}")
            self.curr_episode += 1


def create_default_environment(
    n_agents: int = 3,
    horizon: int = 10,
    max_episodes: int = 1,
    seed: Optional[int] = None
) -> OptimalExecutionEnv:
    """
    Create a default environment with standard parameters for testing.
    
    Args:
        n_agents: Number of agents
        horizon: Episode horizon
        seed: Random seed
        
    Returns:
        Configured OptimalExecutionEnv instance
    """
    # Create utility functions (different targets for each agent)
    utilities = [
        ReserveUtility(i) 
        for i in range(4, 4+n_agents)
    ]
    
    # Create feasibility constraints
    constraints = [
        BoxConstraint(min_pos=0, max_pos=10+(i*5)) 
        for i in range(n_agents)
    ]
    
    # Create transition distribution
    transition = ConstantTransition(mean=0, std=0.5)
    
    # Create initial state
    initial_state = State(
        positions=np.zeros(n_agents),
        price=2.0,
        exogenous_supply=0.0,
        alpha=0.1,
        beta=0.1,
        time=0,
        horizon=horizon
    )

    # Create the naive policy class
    policies = [
        UniformPolicy(utilities[i], constraints[i], i) for i in range(n_agents) 
    ]
    
    # Create environment
    env = OptimalExecutionEnv(
        n_agents=n_agents,
        utility_functions=utilities,
        feasibility_constraints=constraints,
        transition_distribution=transition,
        initial_state=initial_state,
        max_episodes=max_episodes,
        horizon=horizon,
        policies=policies,
        discount_factor=0.99,
    )
    env.run()

if __name__ == "__main__":
    create_default_environment()