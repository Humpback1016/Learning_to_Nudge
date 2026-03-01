# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="NCBF-guided exploration for Isaac Lab environments.")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--max_tipping_angle", type=float, default=15.0, help="Maximum tipping angle for safety (degrees).")

parser.add_argument("--noise_magnitude", type=float, default=0.005, help="Action noise magnitude.")
parser.add_argument("--enable_pinocchio", action="store_true",default=False, help="Enable Pinocchio.")


parser.add_argument("--total_runs", type=int, default=1, help="Total number of exploration runs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=1500, help="Maximum steps per exploration run.")

#scene and random seed
parser.add_argument("--seed", type=int, default=2222, #2222(40obj) 666(20obj, 10obj) video:172, dense video:1016 
                    help="Random seed for reproducibility.")
parser.add_argument("--scene_type", type=str, default="extra_large", choices=["small", "medium", "large", "extra_large"], help="Scene type determining number of objects (small=4, medium=10, large=20, extra_large=40)")

#model and normalizer path
parser.add_argument("--h_obj_normalizer_path", type=str, default="results/local_normalizer", help="Path to the object state normalizer.")
parser.add_argument("--ee_normalizer_path", type=str, default="results/arm_normalizer", help="Path to the end-effector normalizer.")
parser.add_argument("--model_path", type=str, default="boundary_check_results/model_batch_11.pt", help="Path to the trained NCBF model.")

#cbf parameters
parser.add_argument("--history_horizon", type=int, default=6, help="Number of history frames to use for NCBF input.")
parser.add_argument("--cbf_threshold", type=float, default=0.0, help="CBF value threshold for safety.")

#camera and video recording parameters
parser.add_argument("--record_video",action="store_true",default=False,help="Record videos during inference.")
parser.add_argument("--video_length",type=int,default=2500,help="Length of recorded videos (in steps).")
parser.add_argument("--video_interval",type=int,default=1,help="Record video every N runs.",)
parser.add_argument("--video_resolution",type=str,default="1280x720",help="Video resolution (e.g., '1920x1080', '1280x720', '640x480').",)
parser.add_argument("--camera_distance",type=float,default=1.0,help="Camera distance from the scene (meters).",)
parser.add_argument("--camera_height",type=float,default=1.77,help="Camera height above the scene (meters).",)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import gymnasium as gym
import numpy as np
import torch
import omni.log
import random
import math
import json
import datetime
from copy import deepcopy
from time import time
import os
from collections import deque
import cv2
import itertools
"""Rest everything follows."""

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

# import managers
from isaaclab_tasks.utils.config import SafetyConfig, RobotConfig, EnvironmentConfig
from isaaclab_tasks.utils.environment_manager import EnvironmentManager
from isaaclab_tasks.utils.robot_manager import RobotManager
from isaaclab_tasks.utils.geometry_calculator import GeometryCalculator
from isaaclab_tasks.utils.safety_calculator import SafetyCalculator

from collect_data import start_new_trajectory, collect_all_environments_trajectory, end_current_trajectory

from models import NCBF, Normalizer

# Import custom modules
from evaluator import Evaluator
from stats_analyzer import StatsAnalyzer
from data_handler import DataHandler
from models import NCBF, Normalizer

env_manager = EnvironmentManager()
robot_manager = RobotManager()
geometry_calc = GeometryCalculator()
safety_calc = SafetyCalculator()
    
def clear_scene():
    """Clean up any existing prims in the scene"""
    import omni.usd
    from pxr import Usd
    
    stage = omni.usd.get_context().get_stage()
    
    # clean up the environment directory (recursively delete all child Prim)
    env_path = "/World/envs"
    if stage.GetPrimAtPath(env_path).IsValid():
        print(f"cleaning environment: {env_path}")
        stage.RemovePrim(env_path)
    
    print("Scene cleared")

def load_ncbf_model_and_normalizers(model_path, h_obj_normalizer_path, ee_normalizer_path):
    """
    Load NCBF model and normalizers
    
    Args:
        model_path: Path to the NCBF model
        h_obj_normalizer_path: Path to the object state normalizer
        ee_normalizer_path: Path to the end-effector normalizer
        
    Returns:
        ncbf_model: Loaded NCBF model
        h_obj_normalizer: Loaded object state normalizer
        ee_normalizer: Loaded end-effector normalizer
    """
    print(f"Loading NCBF model from {model_path}")
    ncbf_model = NCBF.load(
        path=model_path,
        hiddens=[64, 64],  # hidden layer sizes
        seq_hiddens=[64]   # sequence hidden layer sizes
    )
    
    # (4 features: tilt, rel_x, rel_y, rel_z)
    h_obj_normalizer = Normalizer(input_size=4)
    h_obj_normalizer.load_model(h_obj_normalizer_path)
    
    # (2 features: ee_x, ee_y)
    ee_normalizer = Normalizer(input_size=2)
    ee_normalizer.load_model(ee_normalizer_path)
    
    print("NCBF model and normalizers loaded successfully!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ncbf_model = ncbf_model.to(device)
    ncbf_model.eval() 
    
    return ncbf_model, h_obj_normalizer, ee_normalizer

def main():
    # Set random seed
    seed = args_cli.seed
    print(f"Setting random seed to: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Clear the scene
    clear_scene()
    
    # Load NCBF model and normalizers
    ncbf_model, h_obj_normalizer, ee_normalizer = load_ncbf_model_and_normalizers(
        args_cli.model_path,
        args_cli.h_obj_normalizer_path,
        args_cli.ee_normalizer_path
    )
    
    # Initialize evaluator, data handler and stats analyzer
    evaluator = Evaluator(
        env_manager=env_manager,
        robot_manager=robot_manager,
        geometry_calc=geometry_calc,
        safety_calc=safety_calc,
        ncbf_model=ncbf_model,
        h_obj_normalizer=h_obj_normalizer,
        ee_normalizer=ee_normalizer,
        args=args_cli
    )
    
    data_handler = DataHandler()
    stats_analyzer = StatsAnalyzer()
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "randomize_objects"):
        randomize_params = {"scene_type": args_cli.scene_type}
        env_cfg.events.randomize_objects.params = {"params": randomize_params}
    
    print(f"Using scene type: {args_cli.scene_type}")
    print(f"Number of environments: {args_cli.num_envs}")
    
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'num_envs'):
        env_cfg.scene.num_envs = args_cli.num_envs
    
    # Modify other configurations that may affect environment number
    if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'num_envs'):
        env_cfg.sim.num_envs = args_cli.num_envs
    
    # Modify configurations
    env_cfg.terminations.time_out = None
    
    if "Lift" in args_cli.task:
        # Set resampling time range to a large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # Add goal reaching termination condition, otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    
    # Create environment, set render_mode if video recording is enabled
    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode).unwrapped
    
    # Set safety configuration
    safety_calc.config.max_tipping_angle_degrees = args_cli.max_tipping_angle
    safety_calc.config.use_closeness_cost = False  # enable closeness cost
    safety_calc.config.max_closeness_cost = 5.0   # set max closeness cost
    
    print(f"Safety threshold set to: {args_cli.max_tipping_angle} degrees")
    print(f"Closeness cost enabled with max value: {safety_calc.config.max_closeness_cost}")
    
    # Print velocity related settings
    print(f"Robot velocity settings: min={evaluator.MIN_VELOCITY}, max={evaluator.MAX_VELOCITY}, " + 
          f"change interval={evaluator.VELOCITY_CHANGE_INTERVAL} steps")
    
    # Reset environment
    env.reset()
    
    # Setup camera and video recording
    video_output_dir = "evaluation_videos"
    if args_cli.record_video:
        video_output_dir = evaluator.setup_video_recording(
            env,
            video_output_dir,
            args_cli.camera_distance,
            args_cli.camera_height
        )
        print("Video recording setup completed")
    
    # Start exploration loop
    noise_magnitude = args_cli.noise_magnitude
    total_runs = args_cli.total_runs
    max_steps = args_cli.max_steps
    
    print(f"Starting NCBF-guided exploration in all environments, " + 
          f"running {total_runs} iterations with noise magnitude: {noise_magnitude}")
    
    # Storage for evaluation data from all runs
    all_evaluation_data = []
    
    for run_idx in range(total_runs):
        print(f"===== Starting exploration run {run_idx+1}/{total_runs} =====")
        
        # 1. Reset environment
        env.reset()
        evaluator.reset_object_sequences()
        evaluator.env_velocities.clear()
        evaluator.env_velocity_steps.clear()
        
        # Setup video recording for current run
        video_writers = {}
        if args_cli.record_video and (run_idx + 1) % args_cli.video_interval == 0:
            # Parse resolution parameter
            resolution_str = args_cli.video_resolution
            width, height = map(int, resolution_str.split('x'))
            resolution = (width, height)
            
            video_writers = evaluator.create_video_writers(video_output_dir, run_idx + 1, env.num_envs, resolution=resolution)
            print(f"Video recording enabled for run {run_idx + 1} with resolution {resolution}")
        
        for _ in range(3):
            env.sim.render()
        
        # Adjust end-effector height
        zero_delta_pose = np.zeros(6)
        height_adjust_action = evaluator.pre_process_actions(zero_delta_pose, env.num_envs, env.device, env)
        steps = 50
        
        for step in range(steps):
            step_action = height_adjust_action * (step + 1) / steps
            env.step(step_action)
            env.sim.render()
        
        # 3. Start exploration and compute distance, including video recording
        # Parse resolution parameter
        resolution_str = args_cli.video_resolution
        width, height = map(int, resolution_str.split('x'))
        resolution = (width, height)
        
        # Run evaluation
        all_trajectories, evaluation_data = evaluator.run_evaluation(
            env,
            data_handler,
            max_steps=max_steps,
            noise_magnitude=noise_magnitude,
            video_writers=video_writers,
            video_length=args_cli.video_length,
            target_resolution=resolution
        )
        
        # Store evaluation data for this run
        all_evaluation_data.append(evaluation_data)
        
        # Create run stats
        run_stats = {
            'run_id': run_idx + 1,
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'environments': {}
        }
        
        # Extract environment statistics for this run
        for i, stats in enumerate(evaluation_data['trajectory_stats']):
            env_idx = stats['env_id']
            run_stats['environments'][f'env_{env_idx}'] = stats
        
        # Save statistics for this run
        data_handler.save_run_stats(run_stats, run_idx + 1)
        
        # Close video writers
        if video_writers:
            evaluator.close_video_writers(video_writers)
            print(f"Video recording completed for run {run_idx + 1}")
        
        # Save trajectory data
        base_dir = "final_traj_ours"
        data_handler.end_current_trajectory(save=True, base_dir=base_dir)
        print(f"Trajectory data has been saved to the {base_dir} directory")
    
    # Analyze and save statistics for all runs
    print(f"All {total_runs} exploration runs completed")
    stats_summary = stats_analyzer.analyze_all_runs(all_evaluation_data)
    stats_analyzer.print_stats_summary(stats_summary)
    data_handler.save_summary_stats(stats_summary)
    
    # Close environment
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
