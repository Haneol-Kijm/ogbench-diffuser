import argparse
import os

# Set MUJOCO_GL to egl by default if not set, to avoid OpenGL errors in headless environments
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import imageio
import numpy as np
from tqdm import tqdm

import ogbench
import wandb


def visualize_dataset_trajectory(
    env_name, num_trajectories=5, save_dir="visualizations"
):
    """
    Visualizes trajectories from an OGBench dataset by forcing the environment state
    to match the dataset's qpos and qvel. Saves videos locally and uploads to WandB.
    """

    print(f"Loading environment and dataset: {env_name}...")
    # Load environment and dataset with add_info=True to get qpos/qvel
    env, train_dataset, _ = ogbench.make_env_and_datasets(
        env_name, add_info=True, render_mode="rgb_array"
    )

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize WandB
    run = wandb.init(
        project="ogbench-dataset-viz",
        job_type="visualization",
        name=f"{env_name}-viz",
        config={"env_name": env_name, "num_trajectories": num_trajectories},
    )

    # Dataset structure: flat arrays. We need to identify trajectory boundaries.
    # 'terminals' indicates the end of a trajectory.
    terminals = train_dataset["terminals"]
    # Find indices where trajectories end
    traj_end_indices = np.where(terminals == 1)[0]

    # Start indices are 0 and (end_indices + 1)
    traj_start_indices = np.concatenate(([0], traj_end_indices[:-1] + 1))

    print(f"Found {len(traj_start_indices)} trajectories in the dataset.")

    for i in range(min(num_trajectories, len(traj_start_indices))):
        start_idx = traj_start_indices[i]
        end_idx = traj_end_indices[i]
        length = end_idx - start_idx + 1

        print(f"Visualizing trajectory {i} (Length: {length})...")

        frames = []
        # Reset environment to ensure clean state (though we overwrite it immediately)
        env.reset()

        for t in tqdm(range(start_idx, end_idx + 1), leave=False):
            qpos = train_dataset["qpos"][t]
            qvel = train_dataset["qvel"][t]

            # Force set state
            if "button_states" in train_dataset:
                env.unwrapped.set_state(qpos, qvel, train_dataset["button_states"][t])
            else:
                env.unwrapped.set_state(qpos, qvel)

            # Render
            frame = env.render()
            frames.append(frame)

        # Save locally as MP4
        video_path = os.path.join(save_dir, f"{env_name}_traj_{i}.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Saved local video: {video_path}")

        # Upload to WandB
        # WandB expects (Time, Channel, Height, Width) for video
        wandb_video = wandb.Video(
            np.array(frames).transpose(0, 3, 1, 2),
            fps=30,
            format="mp4",
            caption=f"Trajectory {i}",
        )
        wandb.log({f"trajectory_video": wandb_video})

    print("Visualization complete.")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize OGBench dataset trajectories."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="cube-triple-play-v0",
        help="Name of the OGBench environment.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=3,
        help="Number of trajectories to visualize.",
    )
    args = parser.parse_args()

    visualize_dataset_trajectory(args.env_name, num_trajectories=args.num_trajectories)
