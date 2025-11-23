import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from agents.hiql import HIQLAgent, get_config
from flax.training import checkpoints
from utils.flax_utils import restore_agent

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ogbench


@jax.jit
def sample_high_actions(agent, observations, goals, seed, temperature=1.0):
    high_dist = agent.network.select("high_actor")(
        observations, goals, temperature=temperature
    )
    goal_reps = high_dist.sample(seed=seed)
    # Normalize
    goal_reps = (
        goal_reps
        / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True)
        * jnp.sqrt(goal_reps.shape[-1])
    )
    return goal_reps


@jax.jit
def get_goal_representations(agent, observations, goals):
    return agent.network.select("goal_rep")(
        jnp.concatenate([observations, goals], axis=-1),
        params=agent.network.params,
    )


def get_candidate_states(env, density=2.0):
    maze_map = env.unwrapped.maze_map
    height, width = maze_map.shape
    maze_unit = env.unwrapped._maze_unit
    offset_x = env.unwrapped._offset_x
    offset_y = env.unwrapped._offset_y

    candidates = []
    # Create grid
    xs = np.arange(-offset_x, width * maze_unit - offset_x, 1 / density)
    ys = np.arange(-offset_y, height * maze_unit - offset_y, 1 / density)
    XX, YY = np.meshgrid(xs, ys)
    candidates = np.stack([XX.flatten(), YY.flatten()], axis=1)

    # Filter valid
    valid_candidates = []
    for xy in candidates:
        j = int(round((xy[0] + offset_x) / maze_unit))
        i = int(round((xy[1] + offset_y) / maze_unit))
        if 0 <= i < height and 0 <= j < width:
            if maze_map[i, j] == 0:
                valid_candidates.append(xy)

    return np.array(valid_candidates)


def analyze_hiql_subgoals(env_name, checkpoint_path, seed=42):
    # 1. Load Environment
    print(f"Loading environment: {env_name}")
    env = ogbench.make_env_and_datasets(env_name, env_only=True)

    # 2. Load Agent
    print("Loading HIQL Agent...")
    # Dummy inputs for initialization
    ex_ob = env.observation_space.sample()
    ex_action = env.action_space.sample()
    # Add batch dim
    ex_ob = jnp.array(ex_ob[None])
    ex_action = jnp.array(ex_action[None])

    config = get_config()
    # Ensure config matches training (important params)
    config.hidden_dims = (256, 256, 256)  # Default from hiql.py

    agent = HIQLAgent.create(
        seed=seed, ex_observations=ex_ob, ex_actions=ex_action, config=config
    )

    # Restore checkpoint
    print(f"Restoring agent from {checkpoint_path}")
    restore_dir = os.path.dirname(checkpoint_path)
    filename = os.path.basename(checkpoint_path)
    # Expected format: params_1000000.pkl
    try:
        restore_epoch = int(filename.split("_")[1].split(".")[0])
    except Exception as e:
        print(f"Failed to parse epoch from {filename}: {e}")
        print("Expected format: params_XXXXXX.pkl")
        return

    print(f"Restoring agent from {restore_dir} at epoch {restore_epoch}")
    agent = restore_agent(agent, restore_dir, restore_epoch)

    # 3. Detect Crossroads
    print("Detecting Crossroads...")
    maze_map = env.unwrapped.maze_map
    height, width = maze_map.shape
    crossroads = []

    # Directions: Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(height):
        for j in range(width):
            if maze_map[i, j] == 0:  # Empty space
                open_neighbors = 0
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width and maze_map[ni, nj] == 0:
                        open_neighbors += 1

                if open_neighbors >= 3:
                    # Found a crossroad!
                    xy = env.unwrapped.ij_to_xy((i, j))
                    crossroads.append({"ij": (i, j), "xy": xy})
                    print(f"Found Crossroad at grid ({i}, {j}) -> world {xy}")

    if not crossroads:
        print("No crossroads found! Is the map correct?")
        return

    # 4. Collect Subgoals
    print("Collecting Subgoals...")

    # Prepare plot
    plt.figure(figsize=(10, 10))

    # Generate candidates
    candidates = get_candidate_states(env, density=2.0)
    print(f"Generated {len(candidates)} candidate states.")

    # Draw Maze Walls
    # We can use maze_map to draw rectangles
    maze_unit = env.unwrapped._maze_unit
    offset_x = env.unwrapped._offset_x
    offset_y = env.unwrapped._offset_y

    for i in range(height):
        for j in range(width):
            if maze_map[i, j] == 1:  # Wall
                # Convert to world coordinates
                # ij_to_xy returns center. We need bottom-left for Rectangle?
                # Actually ij_to_xy: x = j * unit - offset, y = i * unit - offset
                # Rectangle needs (x, y) of bottom-left corner.
                # Center is (x, y). Size is unit.
                # So bottom-left is (x - unit/2, y - unit/2)
                cx, cy = env.unwrapped.ij_to_xy((i, j))
                rect = plt.Rectangle(
                    (cx - maze_unit / 2, cy - maze_unit / 2),
                    maze_unit,
                    maze_unit,
                    color="black",
                )
                plt.gca().add_patch(rect)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(crossroads)))

    # Fixed Goal for all agents: Bottom-Left (1, 1)
    goal_ij = (1, 1)
    goal_xy = env.unwrapped.ij_to_xy(goal_ij)

    # Plot Goal (Star) - Once, large and visible
    plt.scatter(
        goal_xy[0],
        goal_xy[1],
        color="red",
        marker="*",
        s=300,
        edgecolors="black",
        zorder=10,
        label="Goal",
    )

    for idx, cr in enumerate(crossroads):
        start_xy = cr["xy"]

        # Reset Env to this state
        # We can't easily 'reset' to a specific state with env.reset() options fully.
        # But we can use set_state or set_xy if available.
        # PointEnv has set_xy.
        env.reset()
        env.unwrapped.set_xy(start_xy)
        env.unwrapped.set_goal(goal_xy=goal_xy)

        # Get Observation
        # HIQL expects: observation (includes goal if GC)
        # But HIQL is usually trained on GC datasets.
        # Let's check agent.sample_high_actions signature or usage.
        # Usually: agent.sample_high_actions(observations, goals)

        obs = env.unwrapped.get_ob()  # (StateDim,)
        goal = env.unwrapped.get_oracle_rep()  # (GoalDim,)

        # Sample 100 subgoals
        # Replicate obs/goal
        batch_size = 100
        obs_batch = jnp.tile(obs, (batch_size, 1))
        goal_batch = jnp.tile(goal, (batch_size, 1))

        # HIQL High-Level Policy Sampling
        rng = jax.random.PRNGKey(idx * 100)
        target_reps = sample_high_actions(agent, obs_batch, goal_batch, seed=rng)

        # Retrieve nearest candidates
        # Compute representations for candidates relative to CURRENT obs
        cand_obs = jnp.tile(obs, (len(candidates), 1))
        cand_goals = jnp.array(candidates)
        cand_reps = get_goal_representations(agent, cand_obs, cand_goals)

        # Find nearest neighbors (max dot product)
        scores = jnp.dot(target_reps, cand_reps.T)
        best_indices = jnp.argmax(scores, axis=1)
        retrieved_subgoals = candidates[best_indices]

        # Plot
        c = colors[idx]
        # Plot Start
        plt.scatter(
            start_xy[0],
            start_xy[1],
            color=c,
            marker="o",
            s=100,
            edgecolors="black",
            label=f"Start {idx}",
        )
        # Plot Subgoals (Small dots)
        plt.scatter(
            retrieved_subgoals[:, 0], retrieved_subgoals[:, 1], color=c, alpha=0.3, s=20
        )

    plt.title(f"HIQL Subgoal Distribution at Crossroads\nEnv: {env_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    # Adjust limits to show full maze
    # x range: [-offset_x, width*unit - offset_x]
    plt.xlim(-offset_x - maze_unit, width * maze_unit - offset_x + maze_unit)
    plt.ylim(-offset_y - maze_unit, height * maze_unit - offset_y + maze_unit)

    save_path = f"hiql_subgoal_analysis_{env_name}.png"
    plt.savefig(save_path)
    print(f"Analysis saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="pointmaze-giant-navigate-v0")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/haneol/ogbench-diffuser/impls/exp/OGBench/Benchmark/sd000_20251122_175241/params_1000000.pkl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze_hiql_subgoals(
        env_name=args.env_name, checkpoint_path=args.checkpoint, seed=args.seed
    )
