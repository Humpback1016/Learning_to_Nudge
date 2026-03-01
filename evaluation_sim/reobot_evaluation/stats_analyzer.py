"""
Statistics analyzer module for NCBF-guided exploration in Isaac Lab environments.
This module handles statistical analysis of evaluation data.
"""

import numpy as np
import datetime
import json
import os

class StatsAnalyzer:
    def __init__(self):
        """Initialize the statistics analyzer"""
        pass
    
    def analyze_run_stats(self, evaluation_data):
        """
        Analyze statistics for a single run
        
        Args:
            evaluation_data: Dictionary containing evaluation data
            
        Returns:
            stats_summary: Dictionary containing statistical summary
        """
        if 'trajectory_stats' not in evaluation_data:
            return {}
            
        stats_summary = {}
        
        # Extract basic statistics
        stats_summary['safety_violations'] = evaluation_data.get('safety_violations', 0)
        stats_summary['total_steps'] = evaluation_data.get('total_steps', 0)
        stats_summary['ncbf_preventions'] = evaluation_data.get('ncbf_preventions', 0)
        
        # Calculate violation rate
        if stats_summary['total_steps'] > 0:
            stats_summary['violation_rate'] = stats_summary['safety_violations'] / stats_summary['total_steps'] * 100
        else:
            stats_summary['violation_rate'] = 0
        
        # Analyze trajectory statistics
        trajectory_stats = evaluation_data['trajectory_stats']
        total_trajectories = len(trajectory_stats)
        
        if total_trajectories == 0:
            return stats_summary
            
        # Accumulators for calculating averages
        accumulators = {
            'total_steps': 0,
            'total_path_length': 0.0,
            'total_initial_distance': 0.0,
            'total_final_distance': 0.0,
            'total_max_tilt_angle': 0.0,
            'total_avg_tilt_angle': 0.0,
            'total_unsafe_samples': 0,
            'total_violation_rate': 0.0,
            'targets_reached': 0,
            
            # For safe trajectories
            'safe_traj_count': 0,
            'safe_max_tilt_angle': 0.0,
            
            # For unsafe trajectories
            'unsafe_traj_count': 0,
            'unsafe_max_tilt_angle': 0.0,
            
            # For success and failure trajectory path lengths
            'success_path_length': 0.0,
            'success_count': 0,
            'failure_path_length': 0.0,
            'failure_count': 0,
            
            'safe_and_reached': 0,
            'failure_final_distance': 0.0,
        }
        
        # Process each trajectory
        for stats in trajectory_stats:
            # Accumulate statistics
            accumulators['total_steps'] += stats['steps']
            accumulators['total_path_length'] += stats['path_length']
            accumulators['total_initial_distance'] += stats['initial_distance_to_goal']
            accumulators['total_final_distance'] += stats['final_distance_to_goal']
            accumulators['total_max_tilt_angle'] += stats['max_tilt_angle']
            accumulators['total_avg_tilt_angle'] += stats['avg_tilt_angle']
            accumulators['total_unsafe_samples'] += stats['unsafe_samples']
            accumulators['total_violation_rate'] += stats['violation_rate']
            
            if stats.get('reached_target', False) and stats['unsafe_samples'] == 0:
                accumulators['safe_and_reached'] += 1
            
            # Accumulate to success or failure path lengths based on whether target is reached
            if stats.get('reached_target', False):
                accumulators['targets_reached'] += 1
                accumulators['success_path_length'] += stats['path_length']
                accumulators['success_count'] += 1
            else:
                accumulators['failure_path_length'] += stats['path_length']
                accumulators['failure_count'] += 1
                accumulators['failure_final_distance'] += stats['final_distance_to_goal']
                
            # Separate safe and unsafe trajectories
            if stats['unsafe_samples'] > 0:
                # Unsafe trajectory
                accumulators['unsafe_traj_count'] += 1
                accumulators['unsafe_max_tilt_angle'] += stats['max_tilt_angle']
            else:
                # Safe trajectory
                accumulators['safe_traj_count'] += 1
                accumulators['safe_max_tilt_angle'] += stats['max_tilt_angle']
        
        # Calculate averages
        avg_stats = {}
        if total_trajectories > 0:
            avg_stats['avg_steps'] = accumulators['total_steps'] / total_trajectories
            avg_stats['avg_path_length'] = accumulators['total_path_length'] / total_trajectories
            avg_stats['avg_initial_distance'] = accumulators['total_initial_distance'] / total_trajectories
            avg_stats['avg_final_distance'] = accumulators['total_final_distance'] / total_trajectories
            avg_stats['avg_max_tilt_angle'] = accumulators['total_max_tilt_angle'] / total_trajectories
            avg_stats['avg_avg_tilt_angle'] = accumulators['total_avg_tilt_angle'] / total_trajectories
            avg_stats['avg_unsafe_samples'] = accumulators['total_unsafe_samples'] / total_trajectories
            avg_stats['avg_violation_rate'] = accumulators['total_violation_rate'] / total_trajectories
            avg_stats['target_success_rate'] = (accumulators['targets_reached'] / total_trajectories) * 100
            
            avg_stats['safe_and_reached_rate'] = (accumulators['safe_and_reached'] / total_trajectories) * 100
            
            avg_stats['avg_failure_final_distance'] = accumulators['failure_final_distance'] / accumulators['failure_count'] if accumulators['failure_count'] > 0 else 0.0
            
            # Calculate avg max tilt angles for safe and unsafe trajectories
            avg_stats['avg_max_tilt_angle_safe'] = accumulators['safe_max_tilt_angle'] / accumulators['safe_traj_count'] if accumulators['safe_traj_count'] > 0 else 0.0
            avg_stats['avg_max_tilt_angle_unsafe'] = accumulators['unsafe_max_tilt_angle'] / accumulators['unsafe_traj_count'] if accumulators['unsafe_traj_count'] > 0 else 0.0
            
            # Calculate avg path lengths for success and failure trajectories
            avg_stats['avg_success_path_length'] = accumulators['success_path_length'] / accumulators['success_count'] if accumulators['success_count'] > 0 else 0.0
            avg_stats['avg_failure_path_length'] = accumulators['failure_path_length'] / accumulators['failure_count'] if accumulators['failure_count'] > 0 else 0.0
            
            # Add counts
            avg_stats['targets_reached'] = accumulators['targets_reached']
            avg_stats['total_trajectories'] = total_trajectories
            avg_stats['safe_trajectories'] = accumulators['safe_traj_count']
            avg_stats['unsafe_trajectories'] = accumulators['unsafe_traj_count']
            avg_stats['success_count'] = accumulators['success_count']
            avg_stats['failure_count'] = accumulators['failure_count']
            avg_stats['safe_and_reached_count'] = accumulators['safe_and_reached']
        
        stats_summary['avg_stats'] = avg_stats
        
        return stats_summary
    
    def analyze_all_runs(self, all_evaluation_data):
        """
        Analyze statistics for all runs
        
        Args:
            all_evaluation_data: List of dictionaries containing evaluation data for each run
            
        Returns:
            combined_stats: Dictionary containing combined statistical summary
        """
        if not all_evaluation_data:
            return {}
        
        # Initialize combined stats
        combined_stats = {
            'safety_violations': 0,
            'total_steps': 0,
            'ncbf_preventions': 0,
            'trajectory_stats': [],
        }
        
        # Combine data from all runs
        for run_data in all_evaluation_data:
            combined_stats['safety_violations'] += run_data.get('safety_violations', 0)
            combined_stats['total_steps'] += run_data.get('total_steps', 0)
            combined_stats['ncbf_preventions'] += run_data.get('ncbf_preventions', 0)
            
            if 'trajectory_stats' in run_data:
                combined_stats['trajectory_stats'].extend(run_data['trajectory_stats'])
        
        # Analyze combined data
        stats_summary = self.analyze_run_stats(combined_stats)
        
        return stats_summary
    
    def print_stats_summary(self, stats_summary):
        """
        Print statistics summary
        
        Args:
            stats_summary: Dictionary containing statistical summary
        """
        if not stats_summary:
            print("No statistics available to print.")
            return
        
        print(f"total steps: {stats_summary.get('total_steps', 0)}")
        print(f"safety violations: {stats_summary.get('safety_violations', 0)}")
        print(f"NCBF preventions: {stats_summary.get('ncbf_preventions', 0)}")
        
        if stats_summary.get('total_steps', 0) > 0:
            violation_rate = stats_summary.get('violation_rate', 0)
            print(f"violation rate: {violation_rate:.1f}%")
        
        # Print trajectory statistics
        if 'avg_stats' in stats_summary and stats_summary['avg_stats']:
            avg_stats = stats_summary['avg_stats']
            
            print("\nTrajectory Statistics:")
            print("="*60)
            print(f"  safe and reached success rate: {avg_stats.get('safe_and_reached_rate', 0):.1f}%")
            print(f"  average final distance to goal before failure: {avg_stats.get('avg_failure_final_distance', 0):.2f} m")
            print(f"  target success rate: {avg_stats.get('target_success_rate', 0):.1f}%")
            print(f"  safe trajectories: {avg_stats.get('safe_trajectories', 0)} " + 
                  f"({avg_stats.get('safe_trajectories', 0)/avg_stats.get('total_trajectories', 1)*100:.1f}%)")
            print(f"  average final distance to goal: {avg_stats.get('avg_final_distance', 0):.2f} m")    
            print("="*60)

            print(f"  environments that were both safe and reached target: " + 
                  f"{avg_stats.get('safe_and_reached_count', 0)} out of {avg_stats.get('total_trajectories', 0)}")
            
            print("average statistics of all trajectories:")
            print(f"  average steps: {avg_stats.get('avg_steps', 0):.2f}")
            print(f"  average path length: {avg_stats.get('avg_path_length', 0):.2f} m")
            print(f"  average length before success (reached goal): {avg_stats.get('avg_success_path_length', 0):.2f} m")
            print(f"  average length before failure (not reached goal): {avg_stats.get('avg_failure_path_length', 0):.2f} m")
            print(f"  average initial distance to goal: {avg_stats.get('avg_initial_distance', 0):.2f} m")

            print(f"  average max tilt angle (all): {np.degrees(avg_stats.get('avg_max_tilt_angle', 0)):.2f}°")
            print(f"  average max tilt angle (safe trajectories): {np.degrees(avg_stats.get('avg_max_tilt_angle_safe', 0)):.2f}°")
            print(f"  average max tilt angle (unsafe trajectories): {np.degrees(avg_stats.get('avg_max_tilt_angle_unsafe', 0)):.2f}°")
            print(f"  average average tilt angle: {np.degrees(avg_stats.get('avg_avg_tilt_angle', 0)):.2f}°")
            print(f"  average unsafe samples: {avg_stats.get('avg_unsafe_samples', 0):.2f}")
            print(f"  average violation rate: {avg_stats.get('avg_violation_rate', 0):.1f}%")
            print(f"  total trajectories counted: {avg_stats.get('total_trajectories', 0)}")
            print(f"  unsafe trajectories: {avg_stats.get('unsafe_trajectories', 0)} " + 
                  f"({avg_stats.get('unsafe_trajectories', 0)/avg_stats.get('total_trajectories', 1)*100:.1f}%)")
            print(f"  environments that reached target: {avg_stats.get('targets_reached', 0)} out of {avg_stats.get('total_trajectories', 0)}")
            print(f"  environments that failed to reach target: {avg_stats.get('failure_count', 0)} out of {avg_stats.get('total_trajectories', 0)}")