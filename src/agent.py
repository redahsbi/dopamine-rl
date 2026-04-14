"""
Dopaminergic Reinforcement Learning Agent
==========================================
Biologically-inspired Q-Learning agent modeling the Reward Prediction Error (RPE)
hypothesis of dopamine signaling (Schultz et al., 1997).

The core insight: dopaminergic neurons don't fire for rewards themselves,
but for the *difference* between expected and received reward — the RPE:
    δ(t) = r(t) + γ·V(s') - V(s)

This maps directly onto the TD-error in temporal difference learning.
"""

import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional


@dataclass
class NeuralTrace:
    """Records the dopaminergic signal (RPE / TD-error) over time."""
    timestep: int
    state: int
    action: int
    reward: float
    rpe: float          # δ — Reward Prediction Error (dopamine signal)
    v_before: float     # V(s)  — value estimate before update
    v_after: float      # V(s') — value estimate after update


@dataclass
class EpisodeLog:
    total_reward: float
    steps: int
    mean_rpe: float
    rpe_variance: float
    traces: List[NeuralTrace] = field(default_factory=list)


class DopaminergicAgent:
    """
    Q-Learning agent with explicit Reward Prediction Error tracking.

    The RPE (δ) is computed as:
        δ = r + γ · max_a Q(s', a) - Q(s, a)

    Biologically, a positive δ → burst firing of dopamine neurons (better than expected)
    A negative δ → dip in firing (worse than expected)
    δ ≈ 0     → no change (exactly as expected)

    Parameters
    ----------
    n_states    : number of discrete states in the environment
    n_actions   : number of discrete actions
    alpha       : learning rate (synaptic plasticity analog)
    gamma       : discount factor (temporal horizon of prediction)
    epsilon     : exploration rate (stochastic action selection)
    epsilon_decay: rate at which exploration decreases over time
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: Optional[int] = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        if seed is not None:
            np.random.seed(seed)

        # Q-table: analogous to synaptic weights encoding action values
        self.Q = np.zeros((n_states, n_actions))

        # History of dopaminergic signals across all training
        self.rpe_history: List[float] = []
        self.episode_logs: List[EpisodeLog] = []

    def select_action(self, state: int) -> int:
        """ε-greedy policy: explore randomly or exploit learned values."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def compute_rpe(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Compute Reward Prediction Error δ (the dopamine signal).

        δ = r + γ · V(s') - V(s)
        where V(s) = max_a Q(s, a)  (value of current state)
        """
        v_current = self.Q[state, action]
        v_next = 0.0 if done else np.max(self.Q[next_state])
        rpe = reward + self.gamma * v_next - v_current
        return rpe

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> NeuralTrace:
        """
        TD update using RPE — equivalent to Hebbian-like synaptic modification.
        ΔQ(s,a) = α · δ
        """
        rpe = self.compute_rpe(state, action, reward, next_state, done)
        v_before = self.Q[state, action]

        self.Q[state, action] += self.alpha * rpe
        self.rpe_history.append(rpe)

        return NeuralTrace(
            timestep=len(self.rpe_history),
            state=state,
            action=action,
            reward=reward,
            rpe=rpe,
            v_before=v_before,
            v_after=self.Q[state, action],
        )

    def decay_epsilon(self):
        """Gradually reduce exploration — analogous to habit formation."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_value_map(self) -> np.ndarray:
        """State value function V(s) = max_a Q(s,a)."""
        return np.max(self.Q, axis=1)

    def save(self, path: str):
        data = {
            "Q": self.Q.tolist(),
            "rpe_history": self.rpe_history,
            "epsilon": self.epsilon,
            "params": {
                "n_states": self.n_states,
                "n_actions": self.n_actions,
                "alpha": self.alpha,
                "gamma": self.gamma,
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DopaminergicAgent":
        with open(path, "r") as f:
            data = json.load(f)
        agent = cls(**data["params"])
        agent.Q = np.array(data["Q"])
        agent.rpe_history = data["rpe_history"]
        agent.epsilon = data["epsilon"]
        return agent
