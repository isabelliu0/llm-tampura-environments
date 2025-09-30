"""Collect training data for LLM-based planning."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def run_slam_task(
    task_id: int,
    config_file: str,
    data_save_dir: str,
    global_seed: int,
    max_steps: int = 20,
    vis: bool = False,
) -> tuple[bool, str]:
    """Run a single SLAM2D task and collect trajectory data."""
    cmd = [
        "python",
        "run_planner.py",
        f"--config={config_file}",
        f"--global-seed={global_seed}",
        f"--max-steps={max_steps}",
        "--collect-data",
        f"--data-save-dir={data_save_dir}",
    ]
    if vis:
        cmd.append("--vis=1")

    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=720,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode == 0:
            print(f"Task {task_id} completed successfully")
            return True, result.stdout
        else:
            print(f"Task {task_id} failed with return code {result.returncode}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"Task {task_id} timed out after 12 minutes")
        return False, "Task timed out"
    except Exception as e:
        print(f"Task {task_id} failed with exception: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Collect training data from multiple SLAM2D tasks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./env_configs/slam_collect.yml",
        help="Path to SLAM2D config file",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=5,
        help="Number of tasks/trajectories to collect",
    )
    parser.add_argument(
        "--data-save-dir",
        type=str,
        default="./training_data",
        help="Directory to save collected training data",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum steps per task",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base random seed (will increment for each task)",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Enable visualization",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    data_dir = Path(args.data_save_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()

    for task_id in range(args.num_tasks):
        seed = args.base_seed + task_id
        success, message = run_slam_task(
            task_id=task_id,
            config_file=str(config_path),
            data_save_dir=str(data_dir),
            global_seed=seed,
            max_steps=args.max_steps,
            vis=args.vis,
        )
        results.append((task_id, seed, success, message))
        if task_id < args.num_tasks - 1:
            time.sleep(1)

    end_time = time.time()
    total_time = end_time - start_time
    successful_tasks = sum(1 for _, _, success, _ in results if success)
    failed_tasks = args.num_tasks - successful_tasks
    print("DATA COLLECTION SUMMARY")
    print(f"Total tasks: {args.num_tasks}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {failed_tasks}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per task: {total_time / args.num_tasks:.2f} seconds")
    print(f"\nCollected data saved to: {data_dir.absolute()}")


if __name__ == "__main__":
    main()