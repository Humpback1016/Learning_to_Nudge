# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Auto exploration for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--max_tipping_angle", type=float, default=15.0, help="Maximum tipping angle for safety (degrees).")
# parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps per exploration run.")
# parser.add_argument("--total_runs", type=int, default=8, help="Total number of exploration runs.")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps per exploration run.")
parser.add_argument("--total_runs", type=int, default=1, help="Total number of exploration runs.")
parser.add_argument("--noise_magnitude", type=float, default=0.005, help="Action noise magnitude.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

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
from copy import deepcopy
from time import time

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

# Initialize global manager instance
env_manager = EnvironmentManager()
robot_manager = RobotManager()
geometry_calc = GeometryCalculator()
safety_calc = SafetyCalculator()

DEFAULT_PUSH_HEIGHT = 0.19  # push height
OBJECT_RADIUS = 0.05  # object radius
SAMPLING_RADIUS_FACTOR = 1.5 # sampling radius factor   1.5
MOVE_STEP_SIZE = 0.01  # move step size 
SAMPLING_THRESHOLD = 0.05  # reach the sampling point threshold
SAFETY_THRESHOLD_FACTOR = 1.0  # safety threshold factor
BACKUP_SPEED_FACTOR = 0.5  # backup speed factor
MIN_VELOCITY = 0.001  # meter / step 0.001 
MAX_VELOCITY = 0.005  # meter / step 0.005
VELOCITY_CHANGE_INTERVAL = 5  # update velocity every 5 steps

# Note that every 'velocity' is actually: meter / step

# statistic data
exploration_data = {
    'trajectories': [],
    'safety_violations': 0,
    'total_steps': 0
}

# global variables
env_object_sequences = {}  # store the random object access sequence for each environment
env_current_target_idx = {}  # store the index of the current target object in the sequence for each environment
env_velocities = {}  # store the current velocity for each environment
env_velocity_steps = {}  # record the number of steps since the last velocity update for each environment

# statistic data
exploration_data = {
    'trajectories': [],
    'safety_violations': 0,
    'total_steps': 0
}

# Global variables related to the random object access sequence
env_object_sequences = {}  # Stores the random object access sequence for each environment
env_current_target_idx = {}  # Stores the index of the current target object in the sequence for each environment


def reset_object_sequences():
    """Reset object access sequences for all environments"""
    global env_object_sequences, env_current_target_idx
    env_object_sequences.clear()
    env_current_target_idx.clear()
    print("All object access sequences have been reset")

def pre_process_actions(
    delta_pose: np.ndarray,
    num_envs: int,
    device: str,
    env,  # get current EE state
) -> torch.Tensor:
    """
    supports position_zero mode (using only Δx, Δy), but can not keep the end-effector facing downward
    """
    # get current ee state
    ee_pos, _ = robot_manager.get_ee_state(env)
    if ee_pos is None:
        # return 3-dim：Δx, Δy, gripper
        return torch.zeros((num_envs, 3), dtype=torch.float32, device=device)

    #  Δx, Δy
    if len(delta_pose) >= 2:
        delta_xy = delta_pose[:2]
    else:
        delta_xy = [0.0, 0.0]

    delta_xy_tensor = torch.tensor(delta_xy, dtype=torch.float32, device=device).repeat(num_envs, 1)

    # gripper velocity: closed or open
    gripper_vel = torch.full((num_envs, 1), -1.0, dtype=torch.float32, device=device)  # -1.0 代表夹爪关闭

    # output: Δx, Δy, gripper
    return torch.cat([delta_xy_tensor, gripper_vel], dim=1)

def generate_sampling_point(env_idx, object_states_tensor, push_height):
    """
    Generate sampling points around a random object in the specified environment

    Args:
        env_idx: Index of the environment
        object_states_tensor: Tensor of object states
        push_height: Operation height

    Returns:
        Array of sampling points
    """
        
    # Randomly select an object as reference
    obj_idx = random.randint(0, object_states_tensor.shape[1] - 1)
    obj_pos = object_states_tensor[env_idx, obj_idx, :3].cpu().numpy()
    
    # Compute the sampling range around the object
    x_range = [obj_pos[0] - OBJECT_RADIUS*SAMPLING_RADIUS_FACTOR, 
               obj_pos[0] + OBJECT_RADIUS*SAMPLING_RADIUS_FACTOR]
    y_range = [obj_pos[1] - OBJECT_RADIUS*SAMPLING_RADIUS_FACTOR, 
               obj_pos[1] + OBJECT_RADIUS*SAMPLING_RADIUS_FACTOR]
    
    # Randomly generate sampling points
    sample_point = [
        np.random.uniform(low=x_range[0], high=x_range[1]),
        np.random.uniform(low=y_range[0], high=y_range[1]),
        push_height
    ]
    
    return np.array(sample_point)

def get_noisy_action(action, random_magnitude):
    """add noise for actions"""
    if random_magnitude == 0:
        return action
    
    randomness = np.random.normal(loc=0, scale=1, size=action.shape) * random_magnitude
    randomness[2] = 0  # do not add noise in z axis
    noisy_action = action + randomness
    return noisy_action

def auto_explore_all_environments(env, max_steps=500, safe_mode=True, noise_magnitude=0.003):
    """Modified auto exploration function - uses random sampling points and collects data organized by trajectories"""
    global exploration_data, env_velocities, env_velocity_steps
    
    print(f"Starting random point exploration in all {env.num_envs} environments, Safe mode: {safe_mode}")
    
    # 1. Start recording new trajectory
    # Get environment configuration (if available)
    env_config = None
    try:
        env_config = env.task.safe_tabletop_config
    except:
        pass
    
    # Get initial robot position
    init_position = None
    ee_pos, _ = robot_manager.get_ee_state(env)
    if ee_pos is not None:
        init_position = ee_pos[0].cpu().numpy()  # Use the position of the first environment
    
    # Start new trajectory recording
    start_new_trajectory(env_config, init_position)
    
    # Initialize data for each environment
    steps = 0
    all_trajectories = [[] for _ in range(env.num_envs)]
    active_envs = list(range(env.num_envs))
    
    # Store current sampling point for each environment
    sampling_points = {}
    # Record whether the sampling point has been reached
    reached_points = {}
    
    # initialize the velocity and velocity update counter for each environment
    for env_idx in active_envs:
        env_velocities[env_idx] = np.random.uniform(MIN_VELOCITY, MAX_VELOCITY)
        env_velocity_steps[env_idx] = 0
        print(f"Environment {env_idx} initial velocity: {env_velocities[env_idx]:.4f}")
    
    push_height = DEFAULT_PUSH_HEIGHT  # Use default push height
    
    # Get object states for generating sampling points
    object_states_tensor, object_names = env_manager.get_object_states_batch(env)
    
    # Generate initial sampling point for each environment
    for env_idx in active_envs:
        sample_point = generate_sampling_point(env_idx, object_states_tensor, push_height)
        if sample_point is not None:
            sampling_points[env_idx] = sample_point
            reached_points[env_idx] = False
            print(f"Environment {env_idx} initial sampling point: {sample_point}")
        else:
            print(f"Environment {env_idx} cannot generate a sampling point, no available objects")
    
    # Record the previous action for each environment for backing off
    previous_actions = {env_idx: None for env_idx in active_envs}
    
    while steps < max_steps and len(active_envs) > 0:
        # Get end-effector positions for all environments
        ee_positions = robot_manager.get_ee_positions_batch(env)
        if ee_positions is None:
            break
        
        # Update object states (for generating new sampling points)
        object_states_tensor, object_names = env_manager.get_object_states_batch(env)
        
        # Generate actions for each environment
        actions = torch.zeros((env.num_envs, 7), device=env.device)
        
        action_list = []

        for env_idx in active_envs:
            # check if the velocity needs to be updated
            env_velocity_steps[env_idx] += 1
            if env_velocity_steps[env_idx] >= VELOCITY_CHANGE_INTERVAL:
                env_velocities[env_idx] = np.random.uniform(MIN_VELOCITY, MAX_VELOCITY)
                env_velocity_steps[env_idx] = 0
                # print(f"Environment {env_idx} velocity updated to: {env_velocities[env_idx]:.4f}")
            
            current_pos = ee_positions[env_idx].cpu().numpy()
            target_pos = sampling_points[env_idx]
            
            # Compute distance to the sampling point
            direction = target_pos - current_pos
            direction_norm = np.linalg.norm(direction[:2])  # Only consider movement in the XY plane
            
            # Check if reached the sampling point
            if direction_norm < SAMPLING_THRESHOLD:
                # Generate a new sampling point using helper function
                sample_point = generate_sampling_point(env_idx, object_states_tensor, push_height)
                if sample_point is not None:
                    sampling_points[env_idx] = sample_point
                else:
                    print(f"Environment {env_idx} cannot generate a new sampling point, no available objects")
                # sampling point reached, action is 0
                action_list.append(torch.zeros(3, device=env.device))
                continue
            
            # Generate action to move toward the sampling point
            if direction_norm > 0.01:  # If the distance is greater than threshold
                # use the current velocity of the environment instead of the fixed MOVE_STEP_SIZE
                current_velocity = env_velocities[env_idx]
                direction[:2] = direction[:2] / direction_norm * current_velocity  # Normalize and scale
                direction[2] = 0  # Keep Z height fixed
                
                if noise_magnitude > 0:
                    direction = get_noisy_action(direction, noise_magnitude)

                # use pre_process_actions to get shape (1, 3) action
                single_action = pre_process_actions(direction[:2], 1, env.device, env)  # only pass Δx, Δy
                action_list.append(single_action[0])  # shape (3,)

                # Record current action for safe backing off
                previous_actions[env_idx] = single_action[0].clone()
                
                # Record trajectory
                all_trajectories[env_idx].append({
                    'step': steps,
                    'ee_pos': current_pos.copy(),
                    'target_pos': target_pos.copy(),
                    'action': direction.copy(),
                    'env_id': env_idx,
                    'velocity': current_velocity  # record the current velocity value
                })
            else:
                # if the distance is small, do not move
                action_list.append(torch.zeros(3, device=env.device))

        # stack all actions to get shape (num_envs, 3)
        actions = torch.stack(action_list, dim=0)

        # # Safety check and backing off
        if safe_mode:
            # get tilt cost
            tipping_costs, _ = safety_calc.calculate_task_safety_cost_batch(env)
            # get distance cost
            closeness_costs, _ = safety_calc.calculate_current_closeness_cost_batch(env)

            for env_idx in active_envs:
                if tipping_costs[env_idx] > np.radians(args_cli.max_tipping_angle * SAFETY_THRESHOLD_FACTOR):
                    # get the specific cost value for output
                    tipping_value = np.degrees(tipping_costs[env_idx].cpu().item()) if hasattr(tipping_costs, 'cpu') else np.degrees(tipping_costs[env_idx])
                    closeness_value = closeness_costs[env_idx].cpu().item() if hasattr(closeness_costs, 'cpu') else closeness_costs[env_idx]
                    
                    print(f"Environment {env_idx} approaching safety threshold (tilt={tipping_value:.2f}°, closeness={closeness_value:.2f}), backing off")
                     
                    # Use reverse of previous action if exists
                    if previous_actions[env_idx] is not None:
                        actions[env_idx] = -previous_actions[env_idx] * BACKUP_SPEED_FACTOR
                    else:
                        # If not recorded, reverse current action
                        actions[env_idx] = -actions[env_idx] * BACKUP_SPEED_FACTOR
                    
                    # Generate a new sampling point using helper function
                    sample_point = generate_sampling_point(env_idx, object_states_tensor, push_height)
                    if sample_point is not None:
                        sampling_points[env_idx] = sample_point
        
        # Execute actions for all environments
        env.step(actions)
        
        # 2. Collect trajectory data - using new trajectory collection function
        collect_all_environments_trajectory(env)
        
        # Check for safety violations
        is_violated, tipping_costs, closeness_costs, total_costs = safety_calc.is_total_safety_violated_batch(env)
        for env_idx in active_envs[:]:
            if is_violated[env_idx]:
                tipping_value = tipping_costs[env_idx].cpu().item() if hasattr(tipping_costs, 'cpu') else tipping_costs[env_idx]
                closeness_value = closeness_costs[env_idx].cpu().item() if hasattr(closeness_costs, 'cpu') else closeness_costs[env_idx]
                
                print(f"Environment {env_idx} violated safety constraints! Tilt angle: {np.degrees(tipping_value):.2f}°, Closeness: {closeness_value:.2f}")
                exploration_data['safety_violations'] += 1

        steps += 1
        exploration_data['total_steps'] += 1
        
        # Print progress every 10 steps
        if steps % 2 == 0:
            print(f"Step {steps}: {len(active_envs)} environments still exploring")
            
            # print ee height and velocity for each environment
            for env_idx in active_envs:
                full_pos = ee_positions[env_idx].cpu().numpy()
                # print(f"env {env_idx} EE position: x={full_pos[0]:.4f}, y={full_pos[1]:.4f}, z={full_pos[2]:.4f}, velocity={env_velocities[env_idx]:.4f}")

            # calculate and print the distance to the object
            distances_tensor, object_names_list, closest_info = robot_manager.calculate_distances_to_objects_batch(env)
            
            if distances_tensor is not None:
                for env_idx in active_envs:
                    # print(f"\nenv {env_idx} distance info:")
                    for obj_idx, name in enumerate(object_names_list):
                        if obj_idx < distances_tensor.shape[1]:
                            distance = distances_tensor[env_idx, obj_idx].cpu().item()
                            print(f"  - distance {name}: {distance:.4f}m")
                    
                    # print the closest object information
                    if env_idx in closest_info:
                        closest = closest_info[env_idx]
                        # print(f"  closest object: {closest['name']} (distance: {closest['distance']:.4f}m)")
    
    return all_trajectories

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

def print_exploration_stats():
    """Print exploration statistics"""
    print("\n" + "="*60)
    print("Exploration Statistics")
    print("="*60)
    print(f"Total trajectories: {len(exploration_data['trajectories'])}")
    print(f"Total steps: {exploration_data['total_steps']}")
    print(f"Number of safety violations: {exploration_data['safety_violations']}")
    if exploration_data['total_steps'] > 0:
        violation_rate = exploration_data['safety_violations'] / len(exploration_data['trajectories']) * 100
        print(f"Violation rate: {violation_rate:.1f}%")
    print("="*60 + "\n")

def main():
    """Main function to run auto exploration in Isaac Lab environments."""

    clear_scene()
    
    # parse the environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    
    print(f"Number of environments: {args_cli.num_envs}")
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'num_envs'):
        env_cfg.scene.num_envs = args_cli.num_envs
    
    # modify other configurations that may affect the number of environments
    if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'num_envs'):
        env_cfg.sim.num_envs = args_cli.num_envs
    
    # modify the configuration
    env_cfg.terminations.time_out = None

    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    
    # create env
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    
    # set safety configuration
    safety_calc.config.max_tipping_angle_degrees = args_cli.max_tipping_angle
    safety_calc.config.use_closeness_cost = False  # Enable distance safety cost
    safety_calc.config.max_closeness_cost = 5.0   # Set the maximum value for distance safety cost

    print(f"Safety threshold set to: {args_cli.max_tipping_angle} degrees")
    print(f"Closeness cost enabled with max value: {safety_calc.config.max_closeness_cost}")
    
    # print the velocity related settings
    print(f"Robot velocity settings: min={MIN_VELOCITY}, max={MAX_VELOCITY}, change interval={VELOCITY_CHANGE_INTERVAL} steps")
    
    # reset the environment
    env.reset()
    
    # Start the exploration loop
    noise_magnitude = args_cli.noise_magnitude
    total_runs = args_cli.total_runs
    max_steps = args_cli.max_steps
    
    print(f"Starting exploration in all environments, running {total_runs} iterations with noise magnitude: {noise_magnitude}")
    
    for run_idx in range(total_runs):
        print(f"===== Starting exploration run {run_idx+1}/{total_runs} =====")
        
        # 1. env reset
        env.reset()
        reset_object_sequences()
        # reset the velocity variables for all environments
        env_velocities.clear()
        env_velocity_steps.clear()
        
        for _ in range(3):
            env.sim.render()
        
        # 2. Calibrate end-effector height to 0.19
        print(f"Calibrating end-effector height to {DEFAULT_PUSH_HEIGHT}...")

        zero_delta_pose = np.zeros(6)
        height_adjust_action = pre_process_actions(zero_delta_pose, env.num_envs, env.device, env)
        steps = 50

        for step in range(steps):
            step_action = height_adjust_action * (step + 1) / steps
            env.step(step_action)
            env.sim.render()

        # optional: verify the height
        ee_pos, _ = robot_manager.get_ee_state(env)
        if ee_pos is not None:
            heights = [ee_pos[i, 2].item() for i in range(env.num_envs)]
            avg_height = sum(heights) / len(heights)
            print(f"Adjusted average end-effector height: {avg_height:.4f}")
            if abs(avg_height - DEFAULT_PUSH_HEIGHT) > 0.01:
                print("Height deviation is large, adjusting again...")
                height_adjust_action = pre_process_actions(zero_delta_pose, env.num_envs, env.device, env)
                env.step(height_adjust_action)
                env.sim.render()
        
        # 3. begin exploration and distance calculation
        all_trajectories = auto_explore_all_environments(env, max_steps=max_steps, safe_mode=True, noise_magnitude=noise_magnitude)
        
        # 4. save data after the ee height is adjusted to 0.19m (object height is 0.2m)
        for traj in all_trajectories:
            if len(traj) > 0:
                exploration_data['trajectories'].append(traj)

        base_dir = "ncbf_trajectories"
        end_current_trajectory(save=True, base_dir=base_dir)
        print(f"Trajectory data has been saved to the {base_dir} directory")
    
    print(f"All {total_runs} exploration runs completed")
    print_exploration_stats()
    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()