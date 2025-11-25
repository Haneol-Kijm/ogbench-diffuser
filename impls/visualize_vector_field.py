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


PRESETS = {
    "large": {
        "env_name": "pointmaze-large-navigate-v0",
        "tasks": [2, 5],
        "checkpoints": {
            "HIQL": "/home/haneol/ogbench-diffuser/impls/exp/OGBench/Benchmark/sd000_20251124_001204/params_1000000.pkl",
            "HIQL-FM": "/home/haneol/ogbench-diffuser/impls/exp/OGBench/hiql_fm_test/sd000_20251123_212547/params_1000000.pkl",
        },
        "output": "vector_field_comparison.png",
    },
    "giant": {
        "env_name": "pointmaze-giant-navigate-v0",
        "tasks": [1, 3],
        "checkpoints": {
            "HIQL": "/home/haneol/ogbench-diffuser/impls/exp/OGBench/Benchmark/sd000_20251122_175241/params_1000000.pkl",
            "HIQL-FM": "/home/haneol/ogbench-diffuser/impls/exp/OGBench/hiql_fm_test/sd000_20251124_065117/params_1000000.pkl",
        },
        "output": "vector_field_giant_comparison.png",
    },
}


def visualize_vector_field(
    env_name, checkpoints, tasks, output_name, crop=None, agent_names=None
):
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

    # Filter checkpoints based on agent_names if provided
    target_checkpoints = checkpoints
    if agent_names:
        target_checkpoints = {k: v for k, v in checkpoints.items() if k in agent_names}

    for name, path in target_checkpoints.items():
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

    if not agents:
        print("No agents loaded. Exiting.")
        return

    # 3. Prepare Grid for Visualization
    # For streamplot, we need a regular grid.
    # get_candidate_states returns valid points, but we need the full meshgrid for interpolation/streamplot.
    # Let's use get_candidate_states to get the bounds and valid mask, but we'll create a regular grid for streamplot.

    density = 2.0
    candidates, (XX, YY) = get_candidate_states(env, density=density)
    print(f"Generated {len(candidates)} candidate states for vector field.")

    # Dense grid for heatmap
    dense_candidates, (dense_XX, dense_YY) = get_candidate_states(env, density=4.0)
    print(f"Generated {len(dense_candidates)} candidate states for heatmap.")

    # 4. Visualization Loop
    nrows = len(tasks)
    ncols = len(agents)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=150)

    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

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

            # Filter for Crop (Heatmap)
            plot_dense_candidates = dense_candidates
            plot_values = values

            if crop:
                x_min, x_max, y_min, y_max = crop
                mask = (
                    (dense_candidates[:, 0] >= x_min)
                    & (dense_candidates[:, 0] <= x_max)
                    & (dense_candidates[:, 1] >= y_min)
                    & (dense_candidates[:, 1] <= y_max)
                )
                plot_dense_candidates = dense_candidates[mask]
                plot_values = values[mask]

            # Interpolate for heatmap
            cntr = ax.tricontourf(
                plot_dense_candidates[:, 0],
                plot_dense_candidates[:, 1],
                plot_values,
                levels=20,
                cmap="viridis",
                alpha=0.6,
            )

            # Add Colorbar if single plot
            if nrows * ncols == 1:
                fig.colorbar(cntr, ax=ax, label="Value")

            # --- B. Vector Field (Quiver) ---
            # Revert to Quiver as requested, with better styling.
            # We use the 'candidates' grid which is already filtered for valid points.

            obs_batch = jnp.array(candidates)
            goal_batch = jnp.tile(goal_rep, (len(candidates), 1))

            # Sample Latent Subgoals
            rng = jax.random.PRNGKey(42)
            target_reps = agent.sample_high_actions(obs_batch, goal_batch, seed=rng)

            # Retrieve Nearest Physical Subgoals
            retrieval_candidates = candidates

            vectors = []
            chunk_size = 50  # Reduced chunk size to prevent OOM

            for i in range(0, len(candidates), chunk_size):
                s_chunk = obs_batch[i : i + chunk_size]
                z_chunk = target_reps[i : i + chunk_size]

                s_expanded = jnp.expand_dims(s_chunk, 1)
                s_tiled = jnp.tile(s_expanded, (1, len(retrieval_candidates), 1))
                s_flat = s_tiled.reshape(-1, 2)

                w_expanded = jnp.expand_dims(retrieval_candidates, 0)
                w_tiled = jnp.tile(w_expanded, (len(s_chunk), 1, 1))
                w_flat = w_tiled.reshape(-1, 2)

                phi_chunk = get_goal_representations(agent, s_flat, w_flat)
                phi_chunk = phi_chunk.reshape(
                    len(s_chunk), len(retrieval_candidates), -1
                )

                scores = jnp.einsum("bd,bnd->bn", z_chunk, phi_chunk)
                best_indices = jnp.argmax(scores, axis=1)

                best_subgoals = retrieval_candidates[best_indices]
                v_chunk = best_subgoals - s_chunk
                vectors.append(v_chunk)

            vectors = np.concatenate(vectors, axis=0)

            # Filter for Crop (Vector Field)
            plot_candidates = candidates
            plot_vectors = vectors

            if crop:
                x_min, x_max, y_min, y_max = crop
                mask = (
                    (candidates[:, 0] >= x_min)
                    & (candidates[:, 0] <= x_max)
                    & (candidates[:, 1] >= y_min)
                    & (candidates[:, 1] <= y_max)
                )
                plot_candidates = candidates[mask]
                plot_vectors = vectors[mask]

            # Quiver Plot
            # zorder=10 (Top)
            # small head, thin body
            ax.quiver(
                plot_candidates[:, 0],
                plot_candidates[:, 1],
                plot_vectors[:, 0],
                plot_vectors[:, 1],
                color="white",
                scale=None,
                scale_units="xy",
                angles="xy",
                headwidth=3,  # Smaller head
                headlength=4,
                headaxislength=3.5,
                width=0.002,  # Thinner body
                zorder=10,
            )

            # --- C. Plot Maze Walls ---
            # zorder=1 (Middle, above heatmap, below quiver)
            # alpha=0.4 (Semi-transparent)
            for i in range(height):
                for j in range(width):
                    if maze_map[i, j] == 1:
                        cx, cy = env.unwrapped.ij_to_xy((i, j))
                        rect = plt.Rectangle(
                            (cx - maze_unit / 2, cy - maze_unit / 2),
                            maze_unit,
                            maze_unit,
                            facecolor="lightgray",
                            edgecolor="gray",
                            alpha=0.4,
                            zorder=1,
                        )
                        ax.add_patch(rect)

            # --- D. Mark Start and Goal ---
            ax.scatter(
                start_xy[0],
                start_xy[1],
                c="cyan",
                s=200,
                marker="o",
                edgecolors="black",
                label="Start",
                zorder=20,
            )
            ax.scatter(
                goal_xy[0],
                goal_xy[1],
                c="red",
                s=300,
                marker="*",
                edgecolors="black",
                label="Goal",
                zorder=20,
            )

            ax.set_title(f"{agent_name} - Task {task_id}", fontsize=14)

            if crop:
                ax.set_xlim(crop[0], crop[1])
                ax.set_ylim(crop[2], crop[3])
            else:
                ax.set_xlim(
                    -offset_x - maze_unit, width * maze_unit - offset_x + maze_unit
                )
                ax.set_ylim(
                    -offset_y - maze_unit, height * maze_unit - offset_y + maze_unit
                )

            ax.set_aspect("equal")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right")

    plt.tight_layout()
    save_dir = "impls/visualizations"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, output_name)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        type=str,
        default="large",
        choices=["large", "giant"],
        help="Configuration preset",
    )
    parser.add_argument("--env_name", type=str, help="Override environment name")
    parser.add_argument(
        "--tasks", type=int, nargs="+", help="Override task IDs (e.g. --tasks 2 5)"
    )
    parser.add_argument("--ckpt_hiql", type=str, help="Override HIQL checkpoint path")
    parser.add_argument(
        "--ckpt_hiql_fm", type=str, help="Override HIQL-FM checkpoint path"
    )
    parser.add_argument("--output", type=str, help="Override output filename")
    parser.add_argument(
        "--crop", type=float, nargs=4, help="Crop region: x_min x_max y_min y_max"
    )
    parser.add_argument(
        "--agents", type=str, nargs="+", help="Select specific agents to visualize"
    )

    args = parser.parse_args()

    # 1. Load Preset
    config = PRESETS[args.preset].copy()

    # 2. Apply Overrides
    if args.env_name:
        config["env_name"] = args.env_name
    if args.tasks:
        config["tasks"] = args.tasks
    if args.ckpt_hiql:
        config["checkpoints"]["HIQL"] = args.ckpt_hiql
    if args.ckpt_hiql_fm:
        config["checkpoints"]["HIQL-FM"] = args.ckpt_hiql_fm
    if args.output:
        config["output"] = args.output

    visualize_vector_field(
        config["env_name"],
        config["checkpoints"],
        config["tasks"],
        config["output"],
        crop=args.crop,
        agent_names=args.agents,
    )
