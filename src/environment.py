"""
Pavlovian Grid Environment
==========================
A gridworld designed to replicate classic conditioning paradigms used
in dopamine neuroscience experiments (Schultz, Dayan & Montague 1997).

The environment includes:
- Conditioned stimuli (CS): neutral states that predict reward
- Unconditioned stimuli (US): reward delivery states
- Neutral zones: uninformative states

This allows us to observe the temporal transfer of the dopamine signal
from US → CS as learning progresses — the hallmark of TD learning
and a key prediction of the RPE hypothesis.
"""

import numpy as np
from typing import Tuple, Optional


class PavlovianGrid:
    """
    N×N gridworld where an agent must navigate from start to goal.

    State encoding: s = row * N + col  (flattened 2D index)
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

    Special zones
    -------------
    - START : top-left corner
    - GOAL  : bottom-right corner  → large positive reward
    - TRAP  : a penalty zone       → negative reward
    - CS    : a cell adjacent to goal that becomes a predictor

    As training progresses, the agent learns to anticipate reward
    at the CS location — mirroring what dopamine neurons do in vivo.
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    def __init__(self, size: int = 5, seed: Optional[int] = 42):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.rng = np.random.default_rng(seed)

        # Key locations
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.trap = (size - 1, 0)
        self.cs_zone = (size - 1, size - 2)   # conditioned stimulus cell

        self.state = self._pos_to_state(self.start)
        self.steps = 0
        self.max_steps = size * size * 4

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.size)

    def reset(self) -> int:
        self.state = self._pos_to_state(self.start)
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Execute action, return (next_state, reward, done, info).

        Reward structure mirrors conditioning experiments:
        - Goal   : +10  (unconditioned stimulus / reward delivery)
        - CS zone: +0.5 (conditioned stimulus — only significant after learning)
        - Trap   : -5   (punishment)
        - Step   : -0.1 (small cost encourages efficient paths)
        """
        row, col = self._state_to_pos(self.state)
        dr, dc = self.ACTIONS[action]
        new_row = np.clip(row + dr, 0, self.size - 1)
        new_col = np.clip(col + dc, 0, self.size - 1)
        new_pos = (new_row, new_col)

        next_state = self._pos_to_state(new_pos)
        self.steps += 1

        # Reward function
        if new_pos == self.goal:
            reward, done = 10.0, True
        elif new_pos == self.trap:
            reward, done = -5.0, False
        elif new_pos == self.cs_zone:
            reward, done = 0.5, False   # weak CS signal
        else:
            reward, done = -0.1, False  # step cost

        if self.steps >= self.max_steps:
            done = True

        self.state = next_state
        info = {
            "position": new_pos,
            "is_goal": new_pos == self.goal,
            "is_trap": new_pos == self.trap,
            "is_cs": new_pos == self.cs_zone,
        }
        return next_state, reward, done, info

    def render_ascii(self, agent_state: Optional[int] = None) -> str:
        """Print ASCII grid for debugging."""
        state = agent_state if agent_state is not None else self.state
        agent_pos = self._state_to_pos(state)
        symbols = {
            self.goal: "G",
            self.trap: "X",
            self.cs_zone: "C",
            self.start: "S",
        }
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                if pos == agent_pos:
                    row.append("A")
                elif pos in symbols:
                    row.append(symbols[pos])
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def get_optimal_path_length(self) -> int:
        """Manhattan distance from start to goal (lower bound on steps)."""
        gr, gc = self.goal
        sr, sc = self.start
        return abs(gr - sr) + abs(gc - sc)
