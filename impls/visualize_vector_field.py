import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from agents.hiql import HIQLAgent
from agents.hiql import get_config as get_hiql_config
from agents.hiql_fm import HIQLFMAgent
from agents.hiql_fm import get_config as get_hiql_fm_config
from utils.flax_utils import restore_agent

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import ogbench


@jax.jit
def get_goal_representations(agent, observations, goals):
    return agent.network.select("goal_rep")(
        jnp.concatenate([observations, goals], axis=-1),
        params=agent.network.params,
    )


@jax.jit
def get_values(agent, observations, goals):
    v1, v2 = agent.network.select("value")(observations, goals)
    return (v1 + v2) / 2


def get_candidate_states(env, density=2.0):
    maze_map = env.unwrapped.maze_map
    height, width = maze_map.shape
    maze_unit = env.unwrapped._maze_unit
    offset_x = env.unwrapped._offset_x
    offset_y = env.unwrapped._offset_y

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

    return np.array(valid_candidates), (XX, YY)


def visualize_vector_field(env_name, checkpoints):
    # 1. Load Environment
    print(f"Loading environment: {env_name}")
    env = ogbench.make_env_and_datasets(env_name, env_only=True)

    # 2. Prepare Agents
    agents = {}
    configs = {
        "HIQL": get_hiql_config(),
        "HIQL-FM": get_hiql_fm_config(),
    }
    agent_classes = {
        "HIQL": HIQLAgent,
        "HIQL-FM": HIQLFMAgent,
    }

    # Dummy inputs for initialization
    ex_ob = env.observation_space.sample()
    ex_action = env.action_space.sample()
    ex_ob = jnp.array(ex_ob[None])
    ex_action = jnp.array(ex_action[None])

    for name, path in checkpoints.items():
        print(f"Loading {name} Agent from {path}...")
        config = configs[name]
        config.hidden_dims = (256, 256, 256)  # Ensure hidden dims match training

        agent = agent_classes[name].create(
            seed=42, ex_observations=ex_ob, ex_actions=ex_action, config=config
        )

        restore_dir = os.path.dirname(path)
        filename = os.path.basename(path)
        try:
            restore_epoch = int(filename.split("_")[1].split(".")[0])
            agent = restore_agent(agent, restore_dir, restore_epoch)
            agents[name] = agent
        except Exception as e:
            print(f"Failed to restore {name}: {e}")
            return

    # 3. Prepare Grid for Visualization
    density = 1.5  # Adjust density for quiver plot clarity
    candidates, (XX, YY) = get_candidate_states(env, density=density)
    print(f"Generated {len(candidates)} candidate states for vector field.")

    # Dense grid for heatmap
    dense_candidates, (dense_XX, dense_YY) = get_candidate_states(env, density=4.0)
    print(f"Generated {len(dense_candidates)} candidate states for heatmap.")

    # 4. Visualization Loop
    tasks = [2, 5]  # Task IDs to visualize
    fig, axes = plt.subplots(len(tasks), len(agents), figsize=(12, 12))

    # Maze info for plotting
    maze_map = env.unwrapped.maze_map
    height, width = maze_map.shape
    maze_unit = env.unwrapped._maze_unit
    offset_x = env.unwrapped._offset_x
    offset_y = env.unwrapped._offset_y

    for row_idx, task_id in enumerate(tasks):
        # Setup Task
        env.reset(options={"task_id": task_id})
        task_info = env.unwrapped.cur_task_info
        start_xy = task_info["init_xy"]
        goal_xy = task_info["goal_xy"]

        # Oracle Goal Representation (Coordinate)
        goal_rep = jnp.array(goal_xy)

        for col_idx, (agent_name, agent) in enumerate(agents.items()):
            ax = axes[row_idx, col_idx]
            print(f"Processing {agent_name} - Task {task_id}...")

            # --- A. Value Heatmap ---
            # Batch inputs
            obs_batch = jnp.array(dense_candidates)  # State is XY
            goal_batch = jnp.tile(goal_rep, (len(dense_candidates), 1))

            # Compute Values
            values = get_values(agent, obs_batch, goal_batch)

            # Interpolate for heatmap
            # We can use tricontourf or griddata. Since we have filtered grid, let's use tricontourf.
            cntr = ax.tricontourf(
                dense_candidates[:, 0],
                dense_candidates[:, 1],
                values,
                levels=20,
                cmap="viridis",
                alpha=0.6,
            )
            # fig.colorbar(cntr, ax=ax) # Optional: Add colorbar

            # --- B. Vector Field ---
            obs_batch = jnp.array(candidates)
            goal_batch = jnp.tile(goal_rep, (len(candidates), 1))

            # Sample Latent Subgoals
            rng = jax.random.PRNGKey(42)
            target_reps = agent.sample_high_actions(obs_batch, goal_batch, seed=rng)

            # Retrieve Nearest Physical Subgoals
            # 1. Encode all candidates as potential subgoals (relative to current state? No, HIQL encodes [s, g])
            # Wait, get_goal_representations takes (agent, observations, goals).
            # Here 'goals' are the candidate states themselves acting as subgoals.
            # And 'observations' are the current states (candidates).
            # So for each state s in candidates, we want to find s' in candidates such that phi(s, s') ~ target_reps.

            # This is expensive: N states x N candidates.
            # Let's optimize: We can just use the same candidates for retrieval.
            # cand_reps = get_goal_representations(agent, obs_batch, obs_batch) # phi(s, s) ? No.

            # We need phi(s, s_candidate) for all s_candidate.
            # But s varies.
            # HIQL retrieval: argmax_w phi(s, w)^T z
            # We need to compute phi(s, w) for all w in candidates.
            # Since N is ~1000, NxN is ~1M, feasible.

            # Let's do it in batches if needed, but 1000x1000 is fine for JAX.
            # cand_obs = jnp.repeat(obs_batch, len(candidates), axis=0) # Too big?
            # cand_goals = jnp.tile(obs_batch, (len(candidates), 1))

            # Actually, we can iterate over each state s to find its subgoal.
            # Or just use a subset of candidates for retrieval to speed up.

            # Let's define retrieval candidates (can be the same as vector field grid)
            retrieval_candidates = candidates

            vectors = []

            # Compute representations for all pairs is heavy.
            # Let's loop over chunks of states.
            chunk_size = 100
            for i in range(0, len(candidates), chunk_size):
                s_chunk = obs_batch[i : i + chunk_size]  # (B, 2)
                z_chunk = target_reps[i : i + chunk_size]  # (B, D)

                # We need phi(s, w) for s in s_chunk, w in retrieval_candidates.
                # Broadcast: s: (B, 1, 2), w: (1, N, 2) -> (B, N, 2)
                s_expanded = jnp.expand_dims(s_chunk, 1)
                s_tiled = jnp.tile(s_expanded, (1, len(retrieval_candidates), 1))
                s_flat = s_tiled.reshape(-1, 2)

                w_expanded = jnp.expand_dims(retrieval_candidates, 0)
                w_tiled = jnp.tile(w_expanded, (len(s_chunk), 1, 1))
                w_flat = w_tiled.reshape(-1, 2)

                # Compute phi(s, w)
                # This might still be heavy (B*N = 100*1000 = 100k). OK.
                phi_chunk = get_goal_representations(agent, s_flat, w_flat)  # (B*N, D)
                phi_chunk = phi_chunk.reshape(
                    len(s_chunk), len(retrieval_candidates), -1
                )  # (B, N, D)

                # Dot product: z . phi
                # z: (B, D), phi: (B, N, D) -> (B, N)
                scores = jnp.einsum("bd,bnd->bn", z_chunk, phi_chunk)
                best_indices = jnp.argmax(scores, axis=1)

                best_subgoals = retrieval_candidates[best_indices]  # (B, 2)

                # Vector: subgoal - state
                v_chunk = best_subgoals - s_chunk
                vectors.append(v_chunk)

            vectors = np.concatenate(vectors, axis=0)

            # Plot Quiver
            ax.quiver(
                candidates[:, 0],
                candidates[:, 1],
                vectors[:, 0],
                vectors[:, 1],
                color="white",
                scale=None,
                scale_units="xy",
                angles="xy",
            )

            # --- C. Plot Maze Walls ---
            for i in range(height):
                for j in range(width):
                    if maze_map[i, j] == 1:
                        cx, cy = env.unwrapped.ij_to_xy((i, j))
                        rect = plt.Rectangle(
                            (cx - maze_unit / 2, cy - maze_unit / 2),
                            maze_unit,
                            maze_unit,
                            color="black",
                        )
                        ax.add_patch(rect)

            # --- D. Mark Start and Goal ---
            ax.scatter(
                start_xy[0],
                start_xy[1],
                c="cyan",
                s=150,
                marker="o",
                edgecolors="black",
                label="Start",
                zorder=10,
            )
            ax.scatter(
                goal_xy[0],
                goal_xy[1],
                c="red",
                s=250,
                marker="*",
                edgecolors="black",
                label="Goal",
                zorder=10,
            )

            ax.set_title(f"{agent_name} - Task {task_id}")
            ax.set_xlim(-offset_x - maze_unit, width * maze_unit - offset_x + maze_unit)
            ax.set_ylim(
                -offset_y - maze_unit, height * maze_unit - offset_y + maze_unit
            )
            ax.set_aspect("equal")
            if row_idx == 0 and col_idx == 0:
                ax.legend()

    plt.tight_layout()
    save_dir = "impls/visualizations"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vector_field_comparison.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    checkpoints = {
        "HIQL": "/home/haneol/ogbench-diffuser/impls/exp/OGBench/Benchmark/sd000_20251124_001204/params_1000000.pkl",
        "HIQL-FM": "/home/haneol/ogbench-diffuser/impls/exp/OGBench/hiql_fm_test/sd000_20251123_212547/params_1000000.pkl",
    }
    visualize_vector_field("pointmaze-large-navigate-v0", checkpoints)
