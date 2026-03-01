"""
Evaluator module for NCBF-guided exploration in Isaac Lab environments.
This module handles the actual evaluation logic and data collection.
"""

import numpy as np
import torch
from collections import deque
import cv2
import os

class Evaluator:
    def __init__(self, env_manager, robot_manager, geometry_calc, safety_calc,
                 ncbf_model, h_obj_normalizer, ee_normalizer, args):
        """
        Initialize the evaluator with managers and models
        
        Args:
            env_manager: Environment manager for handling environment states
            robot_manager: Robot manager for handling robot states and actions
            geometry_calc: Geometry calculator for computing spatial relationships
            safety_calc: Safety calculator for checking safety constraints
            ncbf_model: The Neural Control Barrier Function model
            h_obj_normalizer: Normalizer for object state history
            ee_normalizer: Normalizer for end-effector state
            args: Command line arguments
        """
        self.env_manager = env_manager
        self.robot_manager = robot_manager
        self.geometry_calc = geometry_calc
        self.safety_calc = safety_calc
        self.ncbf_model = ncbf_model
        self.h_obj_normalizer = h_obj_normalizer
        self.ee_normalizer = ee_normalizer
        self.args = args
        
        # Configuration parameters
        self.DEFAULT_PUSH_HEIGHT = 0.19  # push height
        self.MIN_VELOCITY = 0.01  # meter / step 0.01
        self.MAX_VELOCITY = 0.01  # meter / step 0.03
        self.VELOCITY_CHANGE_INTERVAL = 5  # update velocity every 5 steps
        self.GOAL_THRESHOLD = 0.01
        self.TARGET_X_MIN = 0.78
        self.TARGET_X_MAX = 0.80
        self.TARGET_Y_MIN = -0.1
        self.TARGET_Y_MAX = 0.1
        
        # State tracking variables
        self.env_object_sequences = {}  # store the random object access sequence for each environment
        self.env_current_target_idx = {}  # store the index of the current target object in the sequence for each environment
        self.env_velocities = {}  # store the current velocity for each environment
        self.env_velocity_steps = {}  # record the number of steps since the last velocity update for each environment
        self.env_history_windows = {}

    def reset_object_sequences(self):
        """Reset object access sequences for all environments"""
        self.env_object_sequences.clear()
        self.env_current_target_idx.clear()
        self.env_history_windows.clear()
        print("All object access sequences have been reset")
    
    def init_history_windows(self, env, horizon):
        """
        Initialize history windows for all active environments
        
        Args:
            env: The environment object
            horizon: History horizon length
        """
        object_states_tensor, _ = self.env_manager.get_object_states_batch(env)
        active_envs = list(range(env.num_envs))
        
        for env_idx in active_envs:
            self.env_history_windows[env_idx] = {}
            
            for obj_idx in range(object_states_tensor.shape[1]):
                # get object quaternion
                obj_quat = object_states_tensor[env_idx, obj_idx, 3:7].unsqueeze(0)  # extract quaternion and add dimension [1, 4]
                
                # calculate tilt angle using geometry calculator
                obj_tilt = self.geometry_calc.calculate_tilt_angle_from_quaternion_batch(obj_quat)[0].cpu().item()
                
                # get object position
                obj_pos = object_states_tensor[env_idx, obj_idx, :3].cpu().numpy()
                
                # create initial history window (fill window with current state)
                initial_state = [obj_tilt, obj_pos[0], obj_pos[1], obj_pos[2]]
                self.env_history_windows[env_idx][obj_idx] = deque([initial_state] * horizon, maxlen=horizon)
        
        print(f"Initialized history windows for {len(active_envs)} active environments out of {env.num_envs} total")

    def update_history_windows(self, env):
        """
        Update history windows for all active environments
        
        Args:
            env: The environment object
        """
        object_states_tensor, _ = self.env_manager.get_object_states_batch(env)
        active_envs = list(range(env.num_envs))
        
        for env_idx in active_envs:
            if env_idx not in self.env_history_windows:
                continue
                
            for obj_idx in range(object_states_tensor.shape[1]):
                if obj_idx not in self.env_history_windows[env_idx]:
                    continue
                
                # get object quaternion
                obj_quat = object_states_tensor[env_idx, obj_idx, 3:7].unsqueeze(0)  # extract quaternion and add dimension [1, 4]
                
                # calculate tilt angle using geometry calculator
                obj_tilt = self.geometry_calc.calculate_tilt_angle_from_quaternion_batch(obj_quat)[0].cpu().item()
                
                # get object position
                obj_pos = object_states_tensor[env_idx, obj_idx, :3].cpu().numpy()
                
                # update history window
                current_state = [obj_tilt, obj_pos[0], obj_pos[1], obj_pos[2]]
                self.env_history_windows[env_idx][obj_idx].append(current_state)

    def evaluate_ncbf_safety_batch_efficient(self, env_idx, obj_indices, robot_positions):
        """
        Efficient batch evaluation of NCBF safety
        
        Args:
            env_idx: Environment index
            obj_indices: Indices of objects to check
            robot_positions: Robot positions for each object
            
        Returns:
            cbf_values: CBF values for each object
            is_safe: Boolean array indicating safety for each object
        """
        batch_size = len(obj_indices)
        if batch_size == 0:
            return np.array([]), np.array([])
        
        # Prepare batch data
        batch_histories = []
        batch_positions = []
        
        for i in range(batch_size):
            obj_idx = obj_indices[i]
            robot_pos = robot_positions[i]
            history_window = self.env_history_windows[env_idx][obj_idx]
            
            # Process history window
            history_array = np.array(list(history_window))
            base_pos = history_array[0][1:4]
            
            rel_history = []
            for point in history_array:
                obj_tilt = point[0]
                obj_pos = point[1:4]
                rel_pos = obj_pos - base_pos
                rel_history.append([obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]])
            
            rel_history = np.array(rel_history)
            rel_robot_pos = np.array([robot_pos[0], robot_pos[1]]) - base_pos[:2]
            
            # Normalize
            normalized_rel_history = self.h_obj_normalizer(rel_history, update=False)
            normalized_rel_robot_pos = self.ee_normalizer(np.array([rel_robot_pos]), update=False)[0]
            
            batch_histories.append(normalized_rel_history)
            batch_positions.append(normalized_rel_robot_pos)
        
        # Convert to PyTorch tensors
        device = next(self.ncbf_model.parameters()).device
        
        history_tensors = torch.stack([torch.tensor(h, dtype=torch.float32, device=device) for h in batch_histories])
        position_tensors = torch.tensor(batch_positions, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            cbf_values = self.ncbf_model([history_tensors, position_tensors]).cpu().detach().numpy()
            is_safe = cbf_values < self.args.cbf_threshold
                
        return cbf_values, is_safe
    
    def pre_process_actions(self, delta_pose, num_envs, device, env):
        """
        Process delta pose to get action
        
        Args:
            delta_pose: Delta pose (difference in position)
            num_envs: Number of environments
            device: PyTorch device
            env: Environment object
            
        Returns:
            Processed actions for all environments
        """
        # get current ee state
        ee_pos, _ = self.robot_manager.get_ee_state(env)
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
        gripper_vel = torch.full((num_envs, 1), -1.0, dtype=torch.float32, device=device)  # -1.0 means gripper closed

        # output: Δx, Δy, gripper
        return torch.cat([delta_xy_tensor, gripper_vel], dim=1)
    
    def generate_fixed_target_point(self, env_idx, env, push_height):
        """
        Generate a fixed target point for the environment
        
        Args:
            env_idx: Environment index
            env: Environment object
            push_height: Height for pushing
            
        Returns:
            Target world position
        """
        # generate random relative coordinates
        random_x = np.random.uniform(self.TARGET_X_MIN, self.TARGET_X_MAX)
        random_y = np.random.uniform(self.TARGET_Y_MIN, self.TARGET_Y_MAX)
        
        # get environment origin
        env_origin = env.scene.env_origins[env_idx].cpu().numpy()
        
        # calculate absolute world coordinates
        target_world_pos = np.array([
            env_origin[0] + random_x,
            env_origin[1] + random_y,
            push_height  
        ])
        
        # calculate and print relative coordinates
        relative_coords = target_world_pos - np.array([env_origin[0], env_origin[1], 0.0])
        
        print(f"Env {env_idx} - Origin: {env_origin}, Target: {target_world_pos}")
        print(f"Env {env_idx} - Relative target: ({relative_coords[0]:.2f}, {relative_coords[1]:.2f}, {relative_coords[2]:.2f})")
        
        return target_world_pos
    
    def get_noisy_action(self, action, random_magnitude):
        """
        Add noise to an action
        
        Args:
            action: The original action
            random_magnitude: Noise magnitude
            
        Returns:
            Noisy action
        """
        if random_magnitude == 0:
            return action
        
        randomness = np.random.normal(loc=0, scale=1, size=action.shape) * random_magnitude
        randomness[2] = 0  # do not add noise in z axis
        noisy_action = action + randomness
        return noisy_action
    
    def select_action_with_similarity_preference_batch(self, env_idx, current_pos, target_pos, move_step_size=0.01, num_samples=8):
        """
        Select a safe action with preference for similarity to nominal action
        
        Args:
            env_idx: Environment index
            current_pos: Current position
            target_pos: Target position
            move_step_size: Step size for movement
            num_samples: Number of samples for alternative actions
            
        Returns:
            best_action: The best safe action
            is_safe: Boolean indicating if the action is safe
            best_similarity: Similarity to nominal action
        """
        # 1. Calculate nominal action
        nominal_direction = target_pos - current_pos
        direction_norm = np.linalg.norm(nominal_direction[:2])
        
        if direction_norm < 0.001:  # Already very close to target
            return np.zeros(3), True, 1.0
        
        # Normalize direction
        normalized_direction = nominal_direction.copy()
        normalized_direction[:2] = nominal_direction[:2] / direction_norm * move_step_size
        normalized_direction[2] = 0
        
        # 2. Filter objects within 20cm of the end effector
        nearby_objects = []
        if env_idx in self.env_history_windows:
            for obj_idx in self.env_history_windows[env_idx]:
                # Get object history window
                history_window = self.env_history_windows[env_idx][obj_idx]
                if not history_window:
                    continue
                
                # Get latest and earliest state point
                latest_state = list(history_window)[-1]
                earliest_state = list(history_window)[0]
                
                # Calculate position and tilt changes
                position_change = np.linalg.norm(np.array(latest_state[1:4]) - np.array(earliest_state[1:4]))
                tilt_change = abs(latest_state[0] - earliest_state[0])
                
                # Calculate distance to end effector
                obj_pos = np.array([latest_state[1], latest_state[2], latest_state[3]])
                distance_xy = np.linalg.norm(np.array([obj_pos[0], obj_pos[1]]) - 
                                           np.array([current_pos[0], current_pos[1]]))
                
                if distance_xy <= 0.15 or position_change > 0.0 or tilt_change > 0.0:
                    nearby_objects.append(obj_idx)
        
        # If no nearby objects, use nominal action
        if not nearby_objects:
            return normalized_direction, True, 1.0
        
        # 3. Calculate position after nominal action
        next_pos = current_pos + normalized_direction
        
        # 4. Batch evaluate safety of nominal action for nearby objects
        obj_indices = np.array(nearby_objects)
        robot_positions = np.tile(next_pos, (len(nearby_objects), 1))
        
        cbf_values, is_safe_array = self.evaluate_ncbf_safety_batch_efficient(env_idx, obj_indices, robot_positions)
        is_nominal_safe = np.all(is_safe_array)
        
        # If nominal action is safe, use it
        if is_nominal_safe:
            min_nominal_cbf = np.min(cbf_values) if len(cbf_values) > 0 else 0.0
            return normalized_direction, True, 1.0
        
        # 5. Nominal action not safe, generate candidate actions
        candidate_actions = []
        
        # Sample around nominal direction
        angle_step = 2 * np.pi / num_samples
        for i in range(num_samples):
            angle = i * angle_step
            # Create rotation matrix
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            # Rotate nominal direction
            rotated_direction = np.zeros(3)
            rotated_direction[:2] = rotation @ normalized_direction[:2]
            candidate_actions.append(rotated_direction)
        
        # Add actions with different step sizes
        for scale in [0.25, 0.5, 0.75]:
            scaled_action = normalized_direction * scale
            candidate_actions.append(scaled_action)
        
        # 6. Batch evaluate all candidate actions
        # Create batch of all (object, action) combinations
        all_obj_indices = []
        all_robot_positions = []
        action_indices = []  # Record which action each batch item corresponds to
        
        for action_idx, action in enumerate(candidate_actions):
            next_pos = current_pos + action
            for obj_idx in nearby_objects:
                all_obj_indices.append(obj_idx)
                all_robot_positions.append(next_pos)
                action_indices.append(action_idx)
        
        # Batch evaluate safety
        all_cbf_values, all_is_safe = self.evaluate_ncbf_safety_batch_efficient(env_idx, all_obj_indices, all_robot_positions)
        
        # Process results
        safe_actions = []
        safe_cbfs = []
        safe_similarities = []
        
        # Reorganize results, check if each action is safe for all objects
        for action_idx, action in enumerate(candidate_actions):
            # Find all batch items related to this action
            indices = [i for i, a_idx in enumerate(action_indices) if a_idx == action_idx]
            # Check if this action is safe for all objects
            if all(all_is_safe[i] for i in indices):
                # Calculate similarity
                if np.linalg.norm(action[:2]) > 0 and np.linalg.norm(normalized_direction[:2]) > 0:
                    cosine_sim = np.dot(action[:2], normalized_direction[:2]) / (
                        np.linalg.norm(action[:2]) * np.linalg.norm(normalized_direction[:2])
                    )
                else:
                    cosine_sim = 0
                
                safe_actions.append(action)
                min_cbf = min(all_cbf_values[i] for i in indices) if indices else float('inf')
                safe_cbfs.append(min_cbf)
                safe_similarities.append(cosine_sim)
        
        # 7. Select most similar safe action
        if safe_actions:
            # Sort by similarity
            sorted_indices = np.argsort(safe_similarities)[::-1]
            best_action = safe_actions[sorted_indices[0]]
            best_cbf = safe_cbfs[sorted_indices[0]]
            best_similarity = safe_similarities[sorted_indices[0]]
            
            return best_action, True, best_similarity
        else:
            print(f"Env {env_idx}: No safe action found through local sampling")
            return np.zeros(3), False, 0.0
    
    def setup_video_recording(self, env, video_output_dir="evaluation_videos", camera_distance=1.5, camera_height=0.5):
        """
        Setup camera for video recording
        
        Args:
            env: Environment object
            video_output_dir: Directory to save videos
            camera_distance: Camera distance
            camera_height: Camera height
            
        Returns:
            Video output directory
        """
        os.makedirs(video_output_dir, exist_ok=True)
        
        if hasattr(env, 'sim') and hasattr(env.sim, 'set_camera_view'):
            eye = [camera_distance, camera_distance, camera_height]
            target = [-0.2, -0.65, -1.0]
            env.sim.set_camera_view(eye=eye, target=target)
            print(f"Camera view set: eye={eye}, target={target}")
        
        return video_output_dir

    def create_video_writers(self, video_output_dir, run_idx, num_envs, fps=30, resolution=(1280, 720)):
        """
        Create video writers for recording
        
        Args:
            video_output_dir: Directory to save videos
            run_idx: Run index
            num_envs: Number of environments
            fps: Frames per second
            resolution: Video resolution
            
        Returns:
            Dictionary of video writers
        """
        video_writers = {}
        
        for env_idx in range(num_envs):
            video_filename = os.path.join(video_output_dir, f"run_{run_idx}_env_{env_idx}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, resolution)
            
            if video_writer.isOpened():
                video_writers[env_idx] = video_writer
                print(f"Created video writer for env {env_idx}: {video_filename} (resolution: {resolution})")
            else:
                print(f"Failed to create video writer for env {env_idx}")
        
        return video_writers

    def record_video_frames(self, env, video_writers, frame_count, max_frames, target_resolution=(1280, 720)):
        """
        Record video frames
        
        Args:
            env: Environment object
            video_writers: Dictionary of video writers
            frame_count: Current frame count
            max_frames: Maximum frames to record
            target_resolution: Target resolution
        """
        # start to record after 35th frame
        if not video_writers or frame_count >= max_frames or frame_count < 35:
            return
        
        if hasattr(env, 'render') and env.render_mode == "rgb_array":
            rendered_images = env.render()
            
            for env_idx in range(len(video_writers)):
                if env_idx in video_writers:
                    if isinstance(rendered_images, np.ndarray):
                        if len(rendered_images.shape) == 4:
                            frame = rendered_images[env_idx]
                        else:
                            frame = rendered_images
                    else:
                        continue
                    
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    if frame.shape[:2] != (target_resolution[1], target_resolution[0]):
                        frame = cv2.resize(frame, target_resolution)
                    
                    video_writers[env_idx].write(frame)

    def close_video_writers(self, video_writers):
        """
        Close video writers
        
        Args:
            video_writers: Dictionary of video writers
        """
        for env_idx, writer in video_writers.items():
            if writer is not None:
                writer.release()
                print(f"Closed video writer for env {env_idx}")
    
    def run_evaluation(self, env, data_collector, max_steps=500, noise_magnitude=0.003, 
                      video_writers=None, video_length=1500, target_resolution=(1280, 720)):
        """
        Run evaluation with NCBF guidance
        
        Args:
            env: Environment object
            data_collector: Data collector object
            max_steps: Maximum steps per run
            noise_magnitude: Magnitude of noise to add to actions
            video_writers: Video writers for recording
            video_length: Maximum video length
            target_resolution: Video resolution
            
        Returns:
            all_trajectories: List of trajectories for all environments
            evaluation_data: Dictionary containing evaluation data
        """
        # Initialize evaluation data
        evaluation_data = {
            'safety_violations': 0,
            'total_steps': 0,
            'ncbf_preventions': 0,
        }
        
        # Get environment configuration and initial robot position
        try:
            env_config = env.task.safe_tabletop_config
        except:
            env_config = None
        
        ee_pos, _ = self.robot_manager.get_ee_state(env)
        init_position = ee_pos[0].cpu().numpy() if ee_pos is not None else None
        
        # Start new trajectory recording
        data_collector.start_new_trajectory(env_config, init_position)
        
        # Initialize environment data
        steps = 0
        all_trajectories = [[] for _ in range(env.num_envs)]
        
        # Set active environments
        active_envs = list(range(env.num_envs))
        original_active_envs = active_envs.copy()  # Save all initially active environment indices
        print(f"Using {len(active_envs)} active environments out of {env.num_envs} total")
        
        push_height = self.DEFAULT_PUSH_HEIGHT

        # Initialize safety violation counter for each environment
        env_safety_violations = {env_idx: 0 for env_idx in active_envs}
        
        # Target reached status tracking
        reached_targets = {env_idx: False for env_idx in active_envs}
        
        # Initialize sampling points and velocities
        sampling_points = {}
        for env_idx in active_envs:
            # Set initial velocity
            self.env_velocities[env_idx] = np.random.uniform(self.MIN_VELOCITY, self.MAX_VELOCITY)
            self.env_velocity_steps[env_idx] = 0
            
            # Set initial sampling point (fixed target point)
            sampling_points[env_idx] = self.generate_fixed_target_point(env_idx, env, push_height)
            print(f"Env {env_idx}: velocity={self.env_velocities[env_idx]:.4f}, target={sampling_points[env_idx]}")
        
        # Initialize history windows
        self.init_history_windows(env, self.args.history_horizon)

        ee_positions = self.robot_manager.get_ee_positions_batch(env)
        object_states_tensor, _ = self.env_manager.get_object_states_batch(env)
        
        # Initialize data recording
        tilt_history = {env_idx: {obj_idx: [] for obj_idx in range(object_states_tensor.shape[1])} 
                        for env_idx in active_envs}
        path_lengths = {env_idx: 0.0 for env_idx in active_envs}
        last_positions = {env_idx: ee_positions[env_idx].cpu().numpy() for env_idx in active_envs}

        # Main evaluation loop
        while steps < max_steps and active_envs:
            print(f"current step:{steps}")
            # Update environment state
            ee_positions = self.robot_manager.get_ee_positions_batch(env)
            object_states_tensor, _ = self.env_manager.get_object_states_batch(env)
            
            # Only update history windows for active environments
            self.update_history_windows(env)

            if video_writers and steps < video_length:
                self.record_video_frames(env, video_writers, steps, video_length, target_resolution)

            # Check if any environment has reached target
            envs_to_remove = []
            for env_idx in active_envs:
                current_pos = ee_positions[env_idx].cpu().numpy()
                target_pos = sampling_points[env_idx]
                
                # Calculate distance to target
                distance = np.linalg.norm(current_pos[:2] - target_pos[:2])
                
                # If distance is less than threshold, consider it reached target
                if distance <= self.GOAL_THRESHOLD:  
                    print(f"Env {env_idx} reached target! Distance: {distance:.4f}")
                    reached_targets[env_idx] = True
                    envs_to_remove.append(env_idx)
            
            # Remove environments that have reached target
            for env_idx in envs_to_remove:
                active_envs.remove(env_idx)
                # Record final position and path data when environment reaches target
                if env_idx in last_positions:
                    current_pos = ee_positions[env_idx].cpu().numpy()
                    path_lengths[env_idx] += np.linalg.norm(current_pos - last_positions[env_idx])
                    last_positions[env_idx] = current_pos.copy()
                    
                    # Add a final trajectory point
                    target_pos = sampling_points[env_idx]
                    all_trajectories[env_idx].append({
                        'step': steps,
                        'ee_pos': current_pos.copy(),
                        'target_pos': target_pos.copy(),
                        'action': np.zeros(3), 
                        'env_id': env_idx,
                        'velocity': 0.0  
                    })
            
            # If no active environments, end loop early
            if not active_envs:
                print("All active environments reached their targets! Ending current run.")
                break

            # Generate actions for all environments
            action_list = [torch.zeros(3, device=env.device) for _ in range(env.num_envs)]  # default to zero action for all environments
            
            # Only calculate actions for active environments
            for env_idx in active_envs:
                # Only record tilt history for active environments
                quats = object_states_tensor[env_idx, :, 3:7]  # [num_objects, 4]
                tilt_angles = self.geometry_calc.calculate_tilt_angle_from_quaternion_batch(quats)
                
                for obj_idx in range(len(tilt_angles)):
                    tilt_value = tilt_angles[obj_idx].cpu().item()
                    tilt_history[env_idx][obj_idx].append(tilt_value)
                
                # Update path length
                current_pos = ee_positions[env_idx].cpu().numpy()
                if env_idx in last_positions:
                    path_lengths[env_idx] += np.linalg.norm(current_pos - last_positions[env_idx])
                last_positions[env_idx] = current_pos.copy()

                self.env_velocity_steps[env_idx] += 1
                if self.env_velocity_steps[env_idx] >= self.VELOCITY_CHANGE_INTERVAL:
                    self.env_velocities[env_idx] = np.random.uniform(self.MIN_VELOCITY, self.MAX_VELOCITY)
                    self.env_velocity_steps[env_idx] = 0
                
                current_pos = ee_positions[env_idx].cpu().numpy()
                target_pos = sampling_points[env_idx]
                
                # Calculate distance to target
                direction = target_pos - current_pos
                distance = np.linalg.norm(direction[:2])
                
                # Generate action
                if distance > 0.01:
                    current_velocity = self.env_velocities[env_idx]
                    
                    # Use local sampling instead of global map
                    safe_direction, is_safe, cosine_similarity = self.select_action_with_similarity_preference_batch(
                        env_idx, current_pos, target_pos,
                        move_step_size=current_velocity,
                        num_samples=16  # adjust sampling number
                    )

                    # Record NCBF prevention
                    if not is_safe:
                        evaluation_data['ncbf_preventions'] += 1
                        direction = np.zeros(3)  # if no safe action, keep stationary
                    else:
                        direction = safe_direction  # use safe direction
                        # add small noise to increase exploration
                        if noise_magnitude > 0:
                            direction = self.get_noisy_action(direction, noise_magnitude * 0.5)
                    
                    # Process action and record trajectory
                    single_action = self.pre_process_actions(direction[:2], 1, env.device, env)
                    action_list[env_idx] = single_action[0]  # assign action to corresponding environment index
                    
                    all_trajectories[env_idx].append({
                        'step': steps,
                        'ee_pos': current_pos.copy(),
                        'target_pos': target_pos.copy(),
                        'action': direction.copy(),
                        'env_id': env_idx,
                        'velocity': current_velocity
                    })
            
            # Execute actions (all environments, but only active environments have actual actions)
            actions = torch.stack(action_list, dim=0)
            env.step(actions)
            data_collector.collect_all_environments_trajectory(env)
            
            # Check safety violations (only consider active environments)
            is_violated, tipping_costs, closeness_costs, _ = self.safety_calc.is_total_safety_violated_batch(env)
            for env_idx in active_envs:  # only check active environments
                if is_violated[env_idx]:
                    evaluation_data['safety_violations'] += 1
                    env_safety_violations[env_idx] += 1
                    tipping_value = tipping_costs[env_idx].cpu().item() if hasattr(tipping_costs, 'cpu') else tipping_costs[env_idx]
                    print(f"Env {env_idx} safety violation! Tilt: {np.degrees(tipping_value):.2f}°")
            
            # update step count
            steps += 1
            evaluation_data['total_steps'] += 1

        # Calculate statistics for each trajectory
        trajectory_stats = []
        for env_idx in original_active_envs:  # use all original active environments
            if len(all_trajectories[env_idx]) > 0:
                # calculate final distance to target
                final_distance = np.linalg.norm(all_trajectories[env_idx][-1]['ee_pos'][:2] - 
                                               all_trajectories[env_idx][-1]['target_pos'][:2])
                
                # calculate trajectory statistics
                stats = {
                    'env_id': env_idx,
                    'steps': len(all_trajectories[env_idx]),
                    'path_length': path_lengths[env_idx],
                    'initial_distance_to_goal': np.linalg.norm(all_trajectories[env_idx][0]['ee_pos'][:2] - 
                                                              all_trajectories[env_idx][0]['target_pos'][:2]),
                    'final_distance_to_goal': final_distance,
                    'reached_target': reached_targets.get(env_idx, False),
                    'violation_rate': env_safety_violations.get(env_idx, 0) / len(all_trajectories[env_idx]) * 100 if len(all_trajectories[env_idx]) > 0 else 0,
                    'unsafe_samples': env_safety_violations.get(env_idx, 0)
                }
                
                # calculate max and avg tilt angles
                max_tilt_angles = {}
                avg_tilt_angles = {}
                
                if env_idx in tilt_history:
                    for obj_idx in tilt_history[env_idx]:
                        if tilt_history[env_idx][obj_idx]:  # ensure data exists
                            max_tilt_angles[obj_idx] = max(tilt_history[env_idx][obj_idx])
                            avg_tilt_angles[obj_idx] = sum(tilt_history[env_idx][obj_idx]) / len(tilt_history[env_idx][obj_idx])
                
                stats['max_tilt_by_object'] = max_tilt_angles
                stats['avg_tilt_by_object'] = avg_tilt_angles
                stats['max_tilt_angle'] = max(max_tilt_angles.values()) if max_tilt_angles else 0.0
                stats['avg_tilt_angle'] = sum(avg_tilt_angles.values()) / len(avg_tilt_angles) if avg_tilt_angles else 0.0
                
                trajectory_stats.append(stats)
        
        evaluation_data['trajectory_stats'] = trajectory_stats
        
        return all_trajectories, evaluation_data