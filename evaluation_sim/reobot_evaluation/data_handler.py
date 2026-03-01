"""
Data handler module for NCBF-guided exploration in Isaac Lab environments.
This module handles data collection, storage, and loading.
"""

import os
import json
import datetime
import numpy as np

class DataHandler:
    def __init__(self):
        """Initialize the data handler"""
        self.current_trajectory = []
        self.all_trajectories = []
        self.config = None
        self.init_position = None
    
    def start_new_trajectory(self, env_config=None, init_position=None):
        """
        Start a new trajectory recording
        
        Args:
            env_config: Environment configuration
            init_position: Initial robot position
        """
        self.current_trajectory = []
        self.config = env_config
        self.init_position = init_position
        print("Started new trajectory recording")
    
    def collect_all_environments_trajectory(self, env):
        """
        Collect trajectory data for all environments
        
        Args:
            env: Environment object
        """
        # This function would collect trajectory data specific to the environment
        # This implementation is a placeholder since the original implementation is not provided
        pass
    
    def end_current_trajectory(self, save=False, base_dir="final_traj_ours"):
        """
        End current trajectory recording and optionally save it
        
        Args:
            save: Whether to save the trajectory to disk
            base_dir: Directory to save trajectory data
            
        Returns:
            The current trajectory data
        """
        if save:
            self.save_trajectory_data(self.current_trajectory, base_dir)
        
        self.all_trajectories.append(self.current_trajectory)
        print(f"Ended current trajectory recording (length: {len(self.current_trajectory)})")
        return self.current_trajectory
    
    def save_trajectory_data(self, trajectory_data, base_dir="final_traj_ours"):
        """
        Save trajectory data to disk
        
        Args:
            trajectory_data: Trajectory data to save
            base_dir: Directory to save trajectory data
        """
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_dir}/trajectory_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = self._make_serializable(trajectory_data)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        
        print(f"Saved trajectory data to {filename}")
    
    def save_run_stats(self, run_stats, run_idx, base_dir="run_stats_ours_refine_4obj"):
        """
        Save run statistics to disk
        
        Args:
            run_stats: Run statistics to save
            run_idx: Run index
            base_dir: Directory to save statistics
            
        Returns:
            Path to the saved file
        """
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_dir}/run_{run_idx}_stats_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = self._make_serializable(run_stats)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        
        print(f"Saved run {run_idx} statistical data to: {filename}")
        return filename
    
    def save_summary_stats(self, stats_summary, base_dir="run_stats_summary_ours_refine_4obj"):
        """
        Save summary statistics to disk
        
        Args:
            stats_summary: Summary statistics to save
            base_dir: Directory to save statistics
            
        Returns:
            Path to the saved file
        """
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_dir}/summary_stats_{timestamp}.txt"
        
        # Ensure avg_stats exists
        if 'avg_stats' not in stats_summary:
            print("Cannot save summary statistics, avg_stats does not exist")
            return None
        
        stats = stats_summary['avg_stats']
        
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"Safe and reached success rate: {stats['safe_and_reached_rate']:.1f}%\n")
            f.write(f"Average final distance to goal before failure: {stats['avg_failure_final_distance']:.2f} m\n")
            f.write(f"Target success rate: {stats['target_success_rate']:.1f}%\n")
            f.write(f"Safe trajectories: {stats['safe_trajectories']} ({stats['safe_trajectories']/stats['total_trajectories']*100:.1f}%)\n")
            f.write(f"Average final distance to goal: {stats['avg_final_distance']:.2f} m\n")
            f.write("="*60 + "\n")
            
            f.write(f"Environments both safe and reached target: {stats['safe_and_reached_count']} out of {stats['total_trajectories']}\n")
            f.write("Average statistics of all trajectories:\n")
            f.write(f"  Average steps: {stats['avg_steps']:.2f}\n")
            f.write(f"  Average path length: {stats['avg_path_length']:.2f} m\n")
            f.write(f"  Average length before success (reached goal): {stats['avg_success_path_length']:.2f} m\n")
            f.write(f"  Average length before failure (not reached goal): {stats['avg_failure_path_length']:.2f} m\n")
            f.write(f"  Average initial distance to goal: {stats['avg_initial_distance']:.2f} m\n")
            
            f.write(f"  Average max tilt angle (all): {np.degrees(stats['avg_max_tilt_angle']):.2f}\n")
            f.write(f"  Average max tilt angle (safe trajectories): {np.degrees(stats['avg_max_tilt_angle_safe']):.2f}\n")
            f.write(f"  Average max tilt angle (unsafe trajectories): {np.degrees(stats['avg_max_tilt_angle_unsafe']):.2f}\n")
            f.write(f"  Average average tilt angle: {np.degrees(stats['avg_avg_tilt_angle']):.2f}\n")
            f.write(f"  Average unsafe samples: {stats['avg_unsafe_samples']:.2f}\n")
            f.write(f"  Average violation rate: {stats['avg_violation_rate']:.1f}%\n")
            f.write(f"  Total trajectories counted: {stats['total_trajectories']}\n")
            f.write(f"  Unsafe trajectories: {stats['unsafe_trajectories']} ({stats['unsafe_trajectories']/stats['total_trajectories']*100:.1f}%)\n")
            f.write(f"  Environments that reached target: {stats['targets_reached']} out of {stats['total_trajectories']}\n")
            f.write(f"  Environments that failed to reach target: {stats['failure_count']} out of {stats['total_trajectories']}\n")
        
        print(f"Summary statistics saved to: {filename}")
        return filename
    
    def load_run_stats(self, filename):
        """
        Load run statistics from disk
        
        Args:
            filename: Path to the file to load
            
        Returns:
            Loaded run statistics
        """
        try:
            with open(filename, 'r') as f:
                run_stats = json.load(f)
            return run_stats
        except Exception as e:
            print(f"Error loading run statistics from {filename}: {e}")
            return None
    
    def load_trajectory_data(self, filename):
        """
        Load trajectory data from disk
        
        Args:
            filename: Path to the file to load
            
        Returns:
            Loaded trajectory data
        """
        try:
            with open(filename, 'r') as f:
                trajectory_data = json.load(f)
            return trajectory_data
        except Exception as e:
            print(f"Error loading trajectory data from {filename}: {e}")
            return None
    
    def _make_serializable(self, data):
        """
        Make data JSON serializable by converting numpy arrays to lists
        
        Args:
            data: Data to make serializable
            
        Returns:
            Serializable data
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.float32) or isinstance(data, np.float64):
            return float(data)
        elif isinstance(data, np.int32) or isinstance(data, np.int64):
            return int(data)
        elif isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self._make_serializable(item) for item in data]
        else:
            return data