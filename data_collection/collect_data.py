import numpy as np
import os
import time
import torch
# from isaaclab.utils.transformations import transform_position_world_to_ee, transform_position_world_to_obj
from isaaclab_tasks.utils.environment_manager import EnvironmentManager
from isaaclab_tasks.utils.robot_manager import RobotManager
from isaaclab_tasks.utils.geometry_calculator import GeometryCalculator
from isaaclab_tasks.utils.safety_calculator import SafetyCalculator
env_manager = EnvironmentManager()
robot_manager = RobotManager()
geometry_calc = GeometryCalculator()
safety_calc = SafetyCalculator()

# data structure
all_robot_state = []   
all_object_state = [] 
all_safety_cost = [] 
all_is_safe = []
all_robot_joint_states = []  
all_object_names = []  

current_trajectory = {
    'robot_states': [],
    'object_states': [],
    'safety_costs': [],
    'is_safe': [],
    'robot_joint_states': [],  
    'object_names': None  
}

trajectory_configs = []
trajectory_init_positions = []

def start_new_trajectory(env_config=None, init_position=None):
    """start recording new trajectories"""
    global current_trajectory, trajectory_configs, trajectory_init_positions
    
    # reset current trajectory
    current_trajectory = {
        'robot_states': [],
        'object_states': [],
        'safety_costs': [],
        'is_safe': [],
        'robot_joint_states': [],  
        'object_names': None 
    }
    
    if env_config is not None:
        trajectory_configs.append(env_config)
    
    if init_position is not None:
        trajectory_init_positions.append(init_position)
    
    print("start recording new trajectories...")

def end_current_trajectory(save=True, base_dir="trajectories_data"):
    """End the current trajectory and add it to the trajectory list"""
    global current_trajectory, all_robot_state, all_object_state, all_safety_cost, all_is_safe, all_robot_joint_states, all_object_names
    
    # check if trajectory length is greater than 100
    trajectory_length = len(current_trajectory['robot_states'])
    if trajectory_length < 100:
        print(f"Trajectory length ({trajectory_length}) is less than 100, not saving this trajectory")
        return False
    
    all_robot_state.append(current_trajectory['robot_states'])
    all_object_state.append(current_trajectory['object_states'])
    all_safety_cost.append(current_trajectory['safety_costs'])
    all_is_safe.append(current_trajectory['is_safe'])
    all_robot_joint_states.append(current_trajectory['robot_joint_states'])  
    all_object_names.append(current_trajectory['object_names'])  
    
    print(f"Trajectory recording completed, containing {trajectory_length} data points")
    
    if save:
        save_trajectories(base_dir)
    
    return True

def collect_trajectory_data(env, env_idx):
    """collect trajectory data for a single environment, including safety cost for each object and overall safety cost"""
    global current_trajectory
    
    # get ee state
    ee_pos_batch, ee_quat_batch = robot_manager.get_ee_state(env)
    
    # get object state - this function returns position and quaternion, not euler angles
    object_states_tensor, object_names = env_manager.get_object_states_batch(env)
    
    # save object names list (only save once, because object names should be the same for all time steps)
    if current_trajectory['object_names'] is None:
        current_trajectory['object_names'] = object_names
    
    # get robot joint positions
    robot = env.scene.articulations.get("robot", None)
    if robot is not None and hasattr(robot, "data") and hasattr(robot.data, "joint_pos"):
        joint_pos = robot.data.joint_pos[env_idx].cpu().numpy().tolist()
    else:
        joint_pos = []  
        print(f"Warning: Unable to get joint positions for environment {env_idx}")
    
    current_trajectory['robot_joint_states'].append(joint_pos)
    
    # get safety cost
    max_tipping_costs, object_tipping_costs = safety_calc.calculate_task_safety_cost_batch(env)
    
    # get closeness cost
    if safety_calc.config.use_closeness_cost:
        closeness_costs, object_closeness_costs = safety_calc.calculate_current_closeness_cost_batch(env)
    else:
        closeness_costs = torch.zeros_like(max_tipping_costs)
        object_closeness_costs = torch.zeros_like(object_tipping_costs)
    
    # calculate total safety cost (environment level)
    total_costs = max_tipping_costs + closeness_costs
    # calculate total safety cost (object level)
    total_object_costs = object_tipping_costs + object_closeness_costs
    
    # add robot state to trajectory
    robot_state = ee_pos_batch[env_idx].cpu().numpy()
    current_trajectory['robot_states'].append(robot_state)
    
    # add object state to trajectory
    object_states = []
    overall_safety_cost_sum = 0.0
    is_safe_list = []

    for obj_idx in range(object_states_tensor.shape[1]):
        obj_pos = object_states_tensor[env_idx, obj_idx, :3].cpu().numpy()
        obj_quat = object_states_tensor[env_idx, obj_idx, 3:7].cpu().numpy()  
    
        # get object reference, for getting mass and material properties
        obj_name = object_names[obj_idx]
        obj = env.scene.rigid_objects.get(obj_name, None)

        # initialize mass, material properties and COM variables
        obj_mass = None
        obj_material = None
        obj_com = None  # new COM variable

        # get mass, material properties and COM
        if obj is not None and hasattr(obj, "root_physx_view"):
          
            # get mass
            masses = obj.root_physx_view.get_masses()
            if masses is not None and env_idx < masses.shape[0]:
                obj_mass = masses[env_idx].cpu().item() if hasattr(masses[env_idx], 'cpu') else float(masses[env_idx])
            
            # get material properties
            materials = obj.root_physx_view.get_material_properties()
            if materials is not None and env_idx < materials.shape[0]:
                material_tensor = materials[env_idx, 0].cpu() if hasattr(materials[env_idx, 0], 'cpu') else materials[env_idx, 0]
                obj_material = {
                    'static_friction': float(material_tensor[0]),
                    'dynamic_friction': float(material_tensor[1]),
                    'restitution': float(material_tensor[2])
                }
                
            # get COM data
            coms = obj.root_physx_view.get_coms()
            if coms is not None and env_idx < coms.shape[0]:
                com_tensor = coms[env_idx].cpu() if hasattr(coms[env_idx], 'cpu') else coms[env_idx]
                # the first 3 values are xyz coordinates
                obj_com = {
                    'x': float(com_tensor[0]),
                    'y': float(com_tensor[1]),
                    'z': float(com_tensor[2])
                }

        # directly use quaternion to calculate tilt angle
        # here obj_quat is already extracted from object_states_tensor
        obj_quat_tensor = torch.tensor(obj_quat, device=object_states_tensor.device, dtype=object_states_tensor.dtype)
        total_tilt = geometry_calc.calculate_tilt_angle_from_quaternion_batch(obj_quat_tensor.unsqueeze(0)).squeeze().cpu().item()
        
        # get object safety cost
        if obj_idx < object_tipping_costs.shape[1]:
            obj_safety_cost = total_object_costs[env_idx, obj_idx].cpu().item()
            closeness_safety_cost = object_closeness_costs[env_idx, obj_idx].cpu().item()
        else:
            obj_safety_cost = 0.0
            closeness_safety_cost = 0.0
        
        overall_safety_cost_sum += obj_safety_cost
        
        # check if safe
        is_safe = obj_safety_cost <= 0.2618    # 0.2618 
        is_safe_list.append([is_safe])  # each object has a [bool] list
        
        # print(f"env {env_idx}, obj {obj_idx}, obj_safety_cost={obj_safety_cost}")
        
        # combine object state
        obj_state = {
            'position': obj_pos,
            'orientation': obj_quat,
            'tilt_cost': total_tilt,
            'closeness_safety_cost': closeness_safety_cost,
            'safety_cost': obj_safety_cost,
            'overall_safety_cost': overall_safety_cost_sum
        }


        # add mass and material properties (if available)
        if obj_mass is not None:
            obj_state['mass'] = obj_mass
            
        if obj_material is not None:
            obj_state['material'] = obj_material

        if obj_com is not None:
            obj_state['com'] = obj_com

        object_states.append(obj_state)
    
    current_trajectory['object_states'].append(object_states)
    current_trajectory['is_safe'].append(is_safe_list)
    
    safety_cost = total_costs[env_idx].cpu().item()
    current_trajectory['safety_costs'].append(safety_cost)
    
    # print("object_tipping_costs:", object_tipping_costs[env_idx])
    # print("object_closeness_costs:", object_closeness_costs[env_idx])
    
    return True

def collect_all_environments_trajectory(env, active_envs=None):
    """collect trajectory data for active environments"""
    success = False
    
    # if no active environments are provided, assume all environments are active
    if active_envs is None:
        active_envs = range(env.num_envs)
    
    for env_idx in active_envs:
        if collect_trajectory_data(env, env_idx):
            success = True
    
    return success

def save_trajectories(base_dir="trajectories_data"):
    """save trajectory data to JSON file, including ee position and all object states"""
    global all_robot_state, all_object_state, all_safety_cost, all_is_safe, all_robot_joint_states, all_object_names
    
    # create save directory
    os.makedirs(base_dir, exist_ok=True)
    
    # create JSON data structure
    json_data = []
    
    # iterate over all trajectories
    for traj_idx in range(len(all_robot_state)):
        robot_traj = all_robot_state[traj_idx]
        object_traj = all_object_state[traj_idx]
        is_safe_traj = all_is_safe[traj_idx]
        joint_traj = all_robot_joint_states[traj_idx]  
        object_names = all_object_names[traj_idx]  
        
        # ensure trajectory length consistency
        traj_length = min(len(robot_traj), len(object_traj), len(joint_traj))
        # print(f"robot_traj length: {len(robot_traj)}")
        # print(f"object_traj length: {len(object_traj)}")
        # print(f"joint_traj length: {len(joint_traj)}")
        
        # iterate over each time step in the trajectory
        for t in range(traj_length):
            # get robot ee position
            ee_pos = robot_traj[t]  # [x, y, z]
            
            # create ee_state with named fields
            ee_state = {
                "x": float(ee_pos[0]),
                "y": float(ee_pos[1]),
                "z": float(ee_pos[2])
            }
            
            # get joint positions at current time step
            joint_pos = joint_traj[t]
            
            # get all object states
            objects_data = []
            if object_traj[t]:
                for obj_idx, obj in enumerate(object_traj[t]):
                    obj_pos = obj['position']  # [x, y, z]
                    obj_quat = obj['orientation']  # [qx, qy, qz, qw]
                    obj_tilt = obj['tilt_cost']
                    obj_safety_cost = obj['safety_cost']
                    overall_safety_cost = obj['overall_safety_cost']
                    closeness_safety_cost = obj['closeness_safety_cost']
                    
                    obj_name = object_names[obj_idx] if obj_idx < len(object_names) else f"unknown_object_{obj_idx}"
                    
                    # create object state with named fields
                    obj_state = {
                        "name": obj_name,  
                        "position": {
                            "x": float(obj_pos[0]),
                            "y": float(obj_pos[1]),
                            "z": float(obj_pos[2])
                        },
                        "orientation": {
                            "qx": float(obj_quat[0]),
                            "qy": float(obj_quat[1]),
                            "qz": float(obj_quat[2]),
                            "qw": float(obj_quat[3])
                        },
                        "tilt_cost": float(obj_tilt),
                        "safety_cost": float(obj_safety_cost),
                        "overall_safety_cost": float(overall_safety_cost),
                        "closeness_safety_cost": float(closeness_safety_cost)
                    }

                    if 'mass' in obj:
                        obj_state["mass"] = float(obj['mass'])
                    
                    if 'material' in obj:
                        material = obj['material']
                        obj_state["material"] = {
                            "static_friction": float(material['static_friction']),
                            "dynamic_friction": float(material['dynamic_friction']),
                            "restitution": float(material['restitution'])
                        }

                    if 'com' in obj:
                        com = obj['com']
                        obj_state["com"] = {
                            "x": float(com['x']),
                            "y": float(com['y']),
                            "z": float(com['z'])
                        }
                        
                    objects_data.append(obj_state)
            
            # create JSON data entry
            is_safe = is_safe_traj[t] if t < len(is_safe_traj) else None
            data_entry = {
                "ee_state": ee_state,
                "objects_state": objects_data,
                "is_safe": is_safe,
                "robot_joints": [float(j) for j in joint_pos] 
            }
            
            json_data.append(data_entry)
    
    import json
    json_filename = os.path.join(base_dir, "trajectories.json")
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"saved {len(json_data)} data points to {json_filename}")
    
    return True
