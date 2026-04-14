"""
Training & Experiment Runner
=============================
Trains the DopaminergicAgent on PavlovianGrid and records:
- Learning curves (reward per episode)
- RPE dynamics (dopamine signal evolution)
- Value map evolution (how state values develop)
- Temporal transfer of dopamine signal (CS vs US)
"""

import numpy as np
import json
import os
from typing import Dict, List

from agent import DopaminergicAgent
from environment import PavlovianGrid


def run_experiment(
    n_episodes: int = 500,
    grid_size: int = 5,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    seed: int = 42,
    save_dir: str = "../results",
) -> Dict:
    """
    Full training run. Returns a results dict suitable for visualization.
    """
    os.makedirs(save_dir, exist_ok=True)

    env = PavlovianGrid(size=grid_size, seed=seed)
    agent = DopaminergicAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    results = {
        "episode_rewards": [],
        "episode_steps": [],
        "epsilon_values": [],
        "mean_rpe_per_episode": [],
        "rpe_at_cs_per_episode": [],   # RPE when entering CS zone
        "rpe_at_goal_per_episode": [], # RPE when reaching goal
        "value_maps": [],              # snapshots of V(s) over training
        "all_rpe": [],
        "params": {
            "n_episodes": n_episodes,
            "grid_size": grid_size,
            "alpha": alpha,
            "gamma": gamma,
        }
    }

    snapshot_episodes = set(np.linspace(0, n_episodes - 1, 10, dtype=int))

    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_rpe = []
        rpe_cs = []
        rpe_goal = []

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            trace = agent.update(state, action, reward, next_state, done)
            episode_rpe.append(trace.rpe)
            results["all_rpe"].append(trace.rpe)

            if info["is_cs"]:
                rpe_cs.append(trace.rpe)
            if info["is_goal"]:
                rpe_goal.append(trace.rpe)

            episode_reward += reward
            state = next_state

        agent.decay_epsilon()

        results["episode_rewards"].append(episode_reward)
        results["episode_steps"].append(env.steps)
        results["epsilon_values"].append(agent.epsilon)
        results["mean_rpe_per_episode"].append(float(np.mean(episode_rpe)))
        results["rpe_at_cs_per_episode"].append(float(np.mean(rpe_cs)) if rpe_cs else 0.0)
        results["rpe_at_goal_per_episode"].append(float(np.mean(rpe_goal)) if rpe_goal else 0.0)

        if ep in snapshot_episodes:
            results["value_maps"].append({
                "episode": ep,
                "values": agent.get_value_map().tolist(),
            })

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(results["episode_rewards"][-100:])
            print(f"Episode {ep+1:4d}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Avg RPE: {np.mean(episode_rpe):6.3f}")

    # Save agent and results
    agent.save(os.path.join(save_dir, "agent.json"))
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Training complete. Results saved to {save_dir}/")
    print(f"  Final avg reward (last 50 ep): {np.mean(results['episode_rewards'][-50:]):.2f}")
    print(f"  Optimal path length: {env.get_optimal_path_length()} steps")

    return results


if __name__ == "__main__":
    run_experiment(n_episodes=500)
