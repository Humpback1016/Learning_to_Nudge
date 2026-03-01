# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import json
import time
import os
import copy
import itertools
from collections import defaultdict
from tqdm import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Load environment and restore state from JSON.")
parser.add_argument("--num_envs", type=int, default=2000, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--json_file", type=str, default="ncbf_trajectories_0108/trajectories.json", help="Path to JSON file containing state data.")
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio.")
parser.add_argument("--batch_size", type=int, default=2000, help="batch size")
parser.add_argument("--model_path", type=str, default="results_0724/model_iter_20.pt", help="model path")
parser.add_argument("--h_obj_normalizer_path", type=str, default="results_0724/local_normalizer", help="object state normalizer path")
parser.add_argument("--ee_normalizer_path", type=str, default="results_0724/arm_normalizer", help="end-effector normalizer path")
parser.add_argument("--cbf_min", type=float, default=0.00, help="minimum CBF value")
parser.add_argument("--cbf_max", type=float, default=0.015, help="maximum CBF value")
parser.add_argument("--horizon", type=int, default=1, help="history window size")
parser.add_argument("--step", type=int, default=2, help="next step size")
parser.add_argument("--min_seq_len", type=int, default=15, help="minimum sequence length")
parser.add_argument("--output_dir", type=str, default="boundary_check_results", help="output directory")
parser.add_argument("--batch_interval", type=float, default=5.0, help="batch interval time (seconds)")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--training_batch_size", type=int, default=256, help="training batch size")
parser.add_argument("--training_epochs", type=int, default=100, help="training epochs per batch")
parser.add_argument("--margin_threshold", type=float, default=0.05, help="CBF margin threshold")
parser.add_argument("--derivative_threshold", type=float, default=0.02, help="derivative threshold")
parser.add_argument("--best_model_path", type=str, default="best_model.pt", help="best model save path")
parser.add_argument("--buffer_size", type=int, default=10, help="training data buffer size (batch times)")
parser.add_argument("--use_buffer", action="store_true", default=True, help="whether to use training data buffer")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import gymnasium as gym
import omni.log

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

import isaaclab_tasks
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_tasks.utils.environment_manager import EnvironmentManager
from isaaclab_tasks.utils.robot_manager import RobotManager
from isaaclab_tasks.utils.geometry_calculator import GeometryCalculator
from isaaclab_tasks.utils.safety_calculator import SafetyCalculator

from models import NCBF, Normalizer


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class Config:
    """Configuration container"""
    num_envs: int
    task: str
    json_file: str
    model_path: str
    h_obj_normalizer_path: str
    ee_normalizer_path: str
    cbf_min: float
    cbf_max: float
    horizon: int
    step: int
    min_seq_len: int
    output_dir: str
    batch_interval: float
    learning_rate: float
    training_batch_size: int
    training_epochs: int
    margin_threshold: float
    derivative_threshold: float
    best_model_path: str
    buffer_size: int
    use_buffer: bool
    device: str
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        return cls(
            num_envs=args.num_envs,
            task=args.task,
            json_file=args.json_file,
            model_path=args.model_path,
            h_obj_normalizer_path=args.h_obj_normalizer_path,
            ee_normalizer_path=args.ee_normalizer_path,
            cbf_min=args.cbf_min,
            cbf_max=args.cbf_max,
            horizon=args.horizon,
            step=args.step,
            min_seq_len=args.min_seq_len,
            output_dir=args.output_dir,
            batch_interval=args.batch_interval,
            learning_rate=args.learning_rate,
            training_batch_size=args.training_batch_size,
            training_epochs=args.training_epochs,
            margin_threshold=args.margin_threshold,
            derivative_threshold=args.derivative_threshold,
            best_model_path=args.best_model_path,
            buffer_size=args.buffer_size,
            use_buffer=args.use_buffer,
            device=getattr(args, 'device', 'cuda:0')
        )


# ============================================================================
# State Management Classes
# ============================================================================

@dataclass
class TrainingState:
    """Manages training-related state"""
    model: Any
    h_obj_normalizer: Any
    ee_normalizer: Any
    best_performance: float = -float('inf')
    performance_history: List[Dict] = field(default_factory=list)
    
    @classmethod
    def from_paths(cls, model_path: str, h_obj_normalizer_path: str, ee_normalizer_path: str):
        """Load model and normalizers"""
        h_obj_dim = 4
        ee_dim = 2
        
        model = NCBF.load(
            path=model_path,
            hiddens=[64, 64],
            seq_hiddens=[64]
        )
        
        h_obj_normalizer = Normalizer(h_obj_dim)
        ee_normalizer = Normalizer(ee_dim)
        h_obj_normalizer.load_model(h_obj_normalizer_path)
        ee_normalizer.load_model(ee_normalizer_path)
        
        print("NCBF model loaded successfully")
        return cls(model=model, h_obj_normalizer=h_obj_normalizer, ee_normalizer=ee_normalizer)
    
    def save_best_model(self, performance: float, output_dir: str, best_model_path: str):
        """Save model if performance improved"""
        if performance > self.best_performance:
            self.best_performance = performance
            path = os.path.join(output_dir, best_model_path)
            self.model.save(path)
            print(f"New best model saved, performance: {performance:.4f}")
            return True
        return False
    
    def record_performance(self, batch_index: int, performance: float, 
                          safe_samples: int, unsafe_samples: int):
        """Record performance metrics"""
        batch_performance = {
            "batch_index": batch_index,
            "performance": float(performance),
            "safe_samples": safe_samples,
            "unsafe_samples": unsafe_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.performance_history.append(batch_performance)


@dataclass
class DataState:
    """Manages dataset state"""
    safe_local_xs: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    safe_local_nxs: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    safe_ee_xs: np.ndarray = field(default_factory=lambda: np.array([]))
    safe_ee_nxs: np.ndarray = field(default_factory=lambda: np.array([]))
    unsafe_local_xs: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    unsafe_local_nxs: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    unsafe_ee_xs: np.ndarray = field(default_factory=lambda: np.array([]))
    unsafe_ee_nxs: np.ndarray = field(default_factory=lambda: np.array([]))
    
    safe_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    unsafe_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    safe_obj_names: np.ndarray = field(default_factory=lambda: np.array([]))
    unsafe_obj_names: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "safe_local_xs": self.safe_local_xs,
            "safe_local_nxs": self.safe_local_nxs,
            "safe_ee_xs": self.safe_ee_xs,
            "safe_ee_nxs": self.safe_ee_nxs,
            "unsafe_local_xs": self.unsafe_local_xs,
            "unsafe_local_nxs": self.unsafe_local_nxs,
            "unsafe_ee_xs": self.unsafe_ee_xs,
            "unsafe_ee_nxs": self.unsafe_ee_nxs
        }
    
    def get_indices_dict(self) -> Dict:
        """Get indices as dictionary"""
        return {
            "safe_indices": self.safe_indices,
            "unsafe_indices": self.unsafe_indices,
            "safe_obj_names": self.safe_obj_names,
            "unsafe_obj_names": self.unsafe_obj_names
        }
    
    def update_from_relabeling(self, positions_to_move: List[int]):
        """Move samples from unsafe to safe"""
        if not positions_to_move:
            print("No samples to update")
            return
        
        print(f"\n===== Update safety labels =====")
        print(f"Moving {len(positions_to_move)} samples from unsafe to safe")
        
        # Create new lists
        new_safe_local_xs = list(self.safe_local_xs)
        new_safe_local_nxs = list(self.safe_local_nxs)
        new_safe_ee_xs = list(self.safe_ee_xs)
        new_safe_ee_nxs = list(self.safe_ee_nxs)
        new_safe_indices = list(self.safe_indices)
        new_safe_obj_names = list(self.safe_obj_names)
        
        new_unsafe_local_xs = []
        new_unsafe_local_nxs = []
        new_unsafe_ee_xs = []
        new_unsafe_ee_nxs = []
        new_unsafe_indices = []
        new_unsafe_obj_names = []
        
        # Reorganize data
        for i in range(len(self.unsafe_local_xs)):
            if i in positions_to_move:
                # Move to safe
                new_safe_local_xs.append(self.unsafe_local_xs[i])
                if len(self.unsafe_local_nxs) > i:
                    new_safe_local_nxs.append(self.unsafe_local_nxs[i])
                if len(self.unsafe_ee_xs) > i:
                    new_safe_ee_xs.append(self.unsafe_ee_xs[i])
                if len(self.unsafe_ee_nxs) > i:
                    new_safe_ee_nxs.append(self.unsafe_ee_nxs[i])
                new_safe_indices.append(self.unsafe_indices[i])
                new_safe_obj_names.append(self.unsafe_obj_names[i])
            else:
                # Keep in unsafe
                new_unsafe_local_xs.append(self.unsafe_local_xs[i])
                if len(self.unsafe_local_nxs) > i:
                    new_unsafe_local_nxs.append(self.unsafe_local_nxs[i])
                if len(self.unsafe_ee_xs) > i:
                    new_unsafe_ee_xs.append(self.unsafe_ee_xs[i])
                if len(self.unsafe_ee_nxs) > i:
                    new_unsafe_ee_nxs.append(self.unsafe_ee_nxs[i])
                new_unsafe_indices.append(self.unsafe_indices[i])
                new_unsafe_obj_names.append(self.unsafe_obj_names[i])
        
        # Update arrays
        self.safe_local_xs = np.array(new_safe_local_xs, dtype=object)
        self.safe_local_nxs = np.array(new_safe_local_nxs, dtype=object)
        self.safe_ee_xs = np.array(new_safe_ee_xs)
        self.safe_ee_nxs = np.array(new_safe_ee_nxs)
        self.unsafe_local_xs = np.array(new_unsafe_local_xs, dtype=object)
        self.unsafe_local_nxs = np.array(new_unsafe_local_nxs, dtype=object)
        self.unsafe_ee_xs = np.array(new_unsafe_ee_xs)
        self.unsafe_ee_nxs = np.array(new_unsafe_ee_nxs)
        self.safe_indices = np.array(new_safe_indices)
        self.unsafe_indices = np.array(new_unsafe_indices)
        self.safe_obj_names = np.array(new_safe_obj_names)
        self.unsafe_obj_names = np.array(new_unsafe_obj_names)
        
        print(f"Update completed: {len(positions_to_move)} samples moved")
        print(f"Updated data statistics:")
        print(f"  Safe samples: {len(new_safe_local_xs)}")
        print(f"  Unsafe samples: {len(new_unsafe_local_xs)}")


@dataclass
class BatchState:
    """Manages batch processing state"""
    current_index: int = 0
    total_batches: int = 0
    batch_size: int = 0
    results: Dict = field(default_factory=dict)
    boundary_samples: List = field(default_factory=list)
    
    def is_complete(self) -> bool:
        """Check if all batches processed"""
        return self.current_index >= self.total_batches
    
    def next_batch(self):
        """Move to next batch"""
        self.current_index += 1
    
    def get_current_batch_samples(self) -> List:
        """Get samples for current batch"""
        start_idx = self.current_index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.boundary_samples))
        return self.boundary_samples[start_idx:end_idx]
    
    def get_current_batch_indices(self) -> List[int]:
        """Get original indices for current batch"""
        samples = self.get_current_batch_samples()
        return [sample.get("orig_index", i) for i, sample in enumerate(samples)]


@dataclass
class TrainingBuffer:
    """Manages training data buffer"""
    max_size: int
    current_size: int = 0
    safe_local_xs: List = field(default_factory=list)
    safe_local_nxs: List = field(default_factory=list)
    safe_ee_xs: List = field(default_factory=list)
    safe_ee_nxs: List = field(default_factory=list)
    unsafe_local_xs: List = field(default_factory=list)
    unsafe_ee_xs: List = field(default_factory=list)
    safe_indices: List = field(default_factory=list)
    unsafe_indices: List = field(default_factory=list)
    safe_obj_names: List = field(default_factory=list)
    unsafe_obj_names: List = field(default_factory=list)
    
    def add_batch(self, batch_data: Dict, batch_indices: Dict):
        """Add batch data to buffer"""
        # Remove oldest batch if full
        if self.current_size >= self.max_size:
            self._remove_oldest_batch()
        
        # Add new batch
        for key in batch_data:
            buffer_key = key.replace("data_", "")
            if hasattr(self, buffer_key) and len(batch_data[key]) > 0:
                attr = getattr(self, buffer_key)
                if isinstance(attr, list):
                    if isinstance(batch_data[key], np.ndarray):
                        attr.extend(batch_data[key].tolist())
                    else:
                        attr.extend(batch_data[key])
        
        for key in batch_indices:
            if hasattr(self, key) and len(batch_indices[key]) > 0:
                attr = getattr(self, key)
                if isinstance(attr, list):
                    if isinstance(batch_indices[key], np.ndarray):
                        attr.extend(batch_indices[key].tolist())
                    else:
                        attr.extend(batch_indices[key])
        
        self.current_size += 1
        print(f"Current buffer state: {self.current_size}/{self.max_size} batches")
    
    def _remove_oldest_batch(self):
        """Remove oldest batch from buffer"""
        if self.current_size == 0:
            return
        
        batch_len = len(self.safe_local_xs) // self.current_size if self.safe_local_xs else 0
        
        # Remove from all lists
        for attr_name in ['safe_local_xs', 'safe_local_nxs', 'safe_ee_xs', 'safe_ee_nxs',
                         'unsafe_local_xs', 'unsafe_ee_xs', 'safe_indices', 'unsafe_indices',
                         'safe_obj_names', 'unsafe_obj_names']:
            attr = getattr(self, attr_name)
            if attr and batch_len > 0:
                setattr(self, attr_name, attr[batch_len:])
        
        self.current_size -= 1
    
    def to_dict(self) -> Dict:
        """Convert buffer to dictionary"""
        return {
            "safe_local_xs": np.array(self.safe_local_xs, dtype=object),
            "safe_local_nxs": np.array(self.safe_local_nxs, dtype=object),
            "safe_ee_xs": np.array(self.safe_ee_xs),
            "safe_ee_nxs": np.array(self.safe_ee_nxs),
            "unsafe_local_xs": np.array(self.unsafe_local_xs, dtype=object),
            "unsafe_ee_xs": np.array(self.unsafe_ee_xs)
        }
    
    def get_indices_dict(self) -> Dict:
        """Get indices as dictionary"""
        return {
            "safe_indices": np.array(self.safe_indices),
            "unsafe_indices": np.array(self.unsafe_indices),
            "safe_obj_names": np.array(self.safe_obj_names),
            "unsafe_obj_names": np.array(self.unsafe_obj_names)
        }


@dataclass
class EnvironmentState:
    """Manages environment-related state"""
    object_mappings: Dict = field(default_factory=dict)
    snapshot_time_indices: Dict = field(default_factory=dict)
    object_history_buffer: Dict = field(default_factory=dict)
    safe_transitions: List = field(default_factory=list)
    
    def update_object_mapping(self, env_id: int, obj_name: str, mapping_data: Dict):
        """Update object mapping for environment"""
        if env_id not in self.object_mappings:
            self.object_mappings[env_id] = {}
        self.object_mappings[env_id][obj_name] = mapping_data
    
    def get_object_mapping(self, env_id: int, obj_name: str) -> Optional[Dict]:
        """Get object mapping"""
        return self.object_mappings.get(env_id, {}).get(obj_name)


# ============================================================================
# Manager Classes
# ============================================================================

class DataManager:
    """Manages data processing and feature extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.env_manager = EnvironmentManager()
        self.robot_manager = RobotManager()
        self.geometry_calc = GeometryCalculator()
        self.safety_calc = SafetyCalculator()
    
    def load_json_data(self, json_file: str) -> List[Dict]:
        """Load JSON data"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} JSON entries")
        return data
    
    def organize_by_object(self, json_data: List[Dict], max_tilt_angle: float = 0.0, 
                          batch_indices: Optional[List] = None) -> Dict:
        """Organize data by object and filter by tilt angle"""
        if batch_indices is None:
            batch_indices = range(len(json_data))
        
        organized = defaultdict(lambda: defaultdict(list))
        total_samples = 0
        large_tilt_count = 0
        
        for batch_idx, t in enumerate(batch_indices):
            entry = json_data[t]
            objects = entry["objects_state"]
            is_safe = entry.get("is_safe", [[True] for _ in objects])
            ee_state = entry["ee_state"]
            
            for obj_idx, (obj, safe_flag) in enumerate(zip(objects, is_safe)):
                total_samples += 1
                
                tilt_angle_rad = obj.get("tilt_cost", 0)
                max_tilt_rad = np.radians(max_tilt_angle)
                
                if abs(tilt_angle_rad) > max_tilt_rad:
                    large_tilt_count += 1
                    continue
                
                safe = safe_flag[0] if isinstance(safe_flag, list) else safe_flag
                obj_name = obj.get("name", f"object_{obj_idx}")
                
                data_point = {
                    "t": t,
                    "batch_idx": batch_idx,
                    "ee_state": ee_state,
                    "obj_state": obj,
                    "is_safe": safe,
                    "obj_name": obj_name
                }
                organized[obj_idx][t] = data_point
        
        print(f"Total samples: {total_samples}")
        print(f"Filtered (tilt > {max_tilt_angle}°): {large_tilt_count} ({large_tilt_count/total_samples*100:.2f}%)")
        print(f"Retained samples: {total_samples - large_tilt_count}")
        
        return organized
    
    def build_sequences(self, organized: Dict, horizon: int, step: int, 
                       min_seq_len: int) -> Tuple[DataState, Dict]:
        """Build sequence data from organized data"""
        data_state = DataState()
        
        safe_local_xs, safe_local_nxs, safe_ee_xs, safe_ee_nxs = [], [], [], []
        unsafe_local_xs, unsafe_local_nxs, unsafe_ee_xs, unsafe_ee_nxs = [], [], [], []
        safe_indices, unsafe_indices = [], []
        safe_obj_names, unsafe_obj_names = [], []
        
        for obj_idx, time_dict in tqdm(organized.items(), desc="Processing object data"):
            times = sorted(time_dict.keys())
            if len(times) < min_seq_len:
                continue
            
            for i in range(horizon, len(times) - step):
                idxs = times[i-horizon:i]
                next_idxs = times[i-horizon+1:i+1]
                curr_idx = times[i]
                next_idx = times[i+step]
                
                base_idx = idxs[0]
                nbase_idx = next_idxs[0]
                obj_name = time_dict[base_idx]["obj_name"]
                
                # Get base positions
                base_pos = np.array([
                    time_dict[base_idx]["obj_state"]["position"]["x"],
                    time_dict[base_idx]["obj_state"]["position"]["y"],
                    time_dict[base_idx]["obj_state"]["position"]["z"]
                ])
                
                nbase_pos = np.array([
                    time_dict[nbase_idx]["obj_state"]["position"]["x"],
                    time_dict[nbase_idx]["obj_state"]["position"]["y"],
                    time_dict[nbase_idx]["obj_state"]["position"]["z"]
                ])
                
                # Build current state sequence
                local_x = []
                for j in idxs:
                    obj_tilt = time_dict[j]["obj_state"]["tilt_cost"]
                    obj_pos = np.array([
                        time_dict[j]["obj_state"]["position"]["x"],
                        time_dict[j]["obj_state"]["position"]["y"],
                        time_dict[j]["obj_state"]["position"]["z"]
                    ])
                    rel_pos = obj_pos - base_pos
                    features = [obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]]
                    features.append(time_dict[j]["t"])
                    features.append(obj_name)
                    local_x.append(features)
                
                # Build next state sequence
                local_nx = []
                for j in next_idxs:
                    obj_tilt = time_dict[j]["obj_state"]["tilt_cost"]
                    obj_pos = np.array([
                        time_dict[j]["obj_state"]["position"]["x"],
                        time_dict[j]["obj_state"]["position"]["y"],
                        time_dict[j]["obj_state"]["position"]["z"]
                    ])
                    rel_pos = obj_pos - nbase_pos
                    features = [obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]]
                    features.append(time_dict[j]["t"])
                    features.append(obj_name)
                    local_nx.append(features)
                
                # Current and next object positions
                curr_obj_pos = np.array([
                    time_dict[curr_idx]["obj_state"]["position"]["x"],
                    time_dict[curr_idx]["obj_state"]["position"]["y"],
                    time_dict[curr_idx]["obj_state"]["position"]["z"]
                ])
                
                next_obj_pos = np.array([
                    time_dict[next_idx]["obj_state"]["position"]["x"],
                    time_dict[next_idx]["obj_state"]["position"]["y"],
                    time_dict[next_idx]["obj_state"]["position"]["z"]
                ])
                
                # EE positions
                curr_ee_pos = np.array([
                    time_dict[curr_idx]["ee_state"]["x"],
                    time_dict[curr_idx]["ee_state"]["y"]
                ])
                
                next_ee_pos = np.array([
                    time_dict[next_idx]["ee_state"]["x"],
                    time_dict[next_idx]["ee_state"]["y"]
                ])
                
                rel_ee_x = curr_ee_pos - curr_obj_pos[:2]
                rel_ee_nx = next_ee_pos - next_obj_pos[:2]
                
                is_safe = time_dict[curr_idx]["is_safe"]
                orig_index = time_dict[curr_idx]["t"]
                
                # Classify by safety
                if is_safe:
                    safe_local_xs.append(local_x)
                    safe_local_nxs.append(local_nx)
                    safe_ee_xs.append(rel_ee_x)
                    safe_ee_nxs.append(rel_ee_nx)
                    safe_indices.append(orig_index)
                    safe_obj_names.append(obj_name)
                else:
                    unsafe_local_xs.append(local_x)
                    unsafe_local_nxs.append(local_nx)
                    unsafe_ee_xs.append(rel_ee_x)
                    unsafe_ee_nxs.append(rel_ee_nx)
                    unsafe_indices.append(orig_index)
                    unsafe_obj_names.append(obj_name)
        
        # Update data state
        data_state.safe_local_xs = np.array(safe_local_xs, dtype=object)
        data_state.safe_local_nxs = np.array(safe_local_nxs, dtype=object)
        data_state.safe_ee_xs = np.array(safe_ee_xs)
        data_state.safe_ee_nxs = np.array(safe_ee_nxs)
        data_state.unsafe_local_xs = np.array(unsafe_local_xs, dtype=object)
        data_state.unsafe_local_nxs = np.array(unsafe_local_nxs, dtype=object)
        data_state.unsafe_ee_xs = np.array(unsafe_ee_xs)
        data_state.unsafe_ee_nxs = np.array(unsafe_ee_nxs)
        data_state.safe_indices = np.array(safe_indices)
        data_state.unsafe_indices = np.array(unsafe_indices)
        data_state.safe_obj_names = np.array(safe_obj_names)
        data_state.unsafe_obj_names = np.array(unsafe_obj_names)
        
        indices = data_state.get_indices_dict()
        
        return data_state, indices
    
    def extract_training_features(self, data_dict: Dict) -> Dict:
        """Extract training features from data"""
        training_features = {}
        
        # Extract safe sample features
        if "safe_local_xs" in data_dict and len(data_dict["safe_local_xs"]) > 0:
            safe_local_xs = data_dict["safe_local_xs"]
            if isinstance(safe_local_xs, list):
                safe_local_xs = np.array(safe_local_xs, dtype=object)
            
            safe_h_obj_features = np.zeros((len(safe_local_xs), len(safe_local_xs[0]), 4), dtype=np.float32)
            for i in range(len(safe_local_xs)):
                for j in range(len(safe_local_xs[0])):
                    safe_h_obj_features[i, j] = safe_local_xs[i][j][:4].astype(np.float32)
            training_features["safe_h_obj_xs"] = safe_h_obj_features
        
        if "safe_local_nxs" in data_dict and len(data_dict["safe_local_nxs"]) > 0:
            safe_local_nxs = data_dict["safe_local_nxs"]
            if isinstance(safe_local_nxs, list):
                safe_local_nxs = np.array(safe_local_nxs, dtype=object)
            
            safe_h_obj_nxs_features = np.zeros((len(safe_local_nxs), len(safe_local_nxs[0]), 4), dtype=np.float32)
            for i in range(len(safe_local_nxs)):
                for j in range(len(safe_local_nxs[0])):
                    safe_h_obj_nxs_features[i, j] = safe_local_nxs[i][j][:4].astype(np.float32)
            training_features["safe_h_obj_nxs"] = safe_h_obj_nxs_features
        
        if "safe_ee_xs" in data_dict and len(data_dict["safe_ee_xs"]) > 0:
            safe_ee_xs = data_dict["safe_ee_xs"]
            if isinstance(safe_ee_xs, list):
                safe_ee_xs = np.array(safe_ee_xs, dtype=np.float32)
            training_features["safe_ee_xs"] = safe_ee_xs
        
        if "safe_ee_nxs" in data_dict and len(data_dict["safe_ee_nxs"]) > 0:
            safe_ee_nxs = data_dict["safe_ee_nxs"]
            if isinstance(safe_ee_nxs, list):
                safe_ee_nxs = np.array(safe_ee_nxs, dtype=np.float32)
            training_features["safe_ee_nxs"] = safe_ee_nxs
        
        # Extract unsafe sample features
        if "unsafe_local_xs" in data_dict and len(data_dict["unsafe_local_xs"]) > 0:
            unsafe_local_xs = data_dict["unsafe_local_xs"]
            if isinstance(unsafe_local_xs, list):
                unsafe_local_xs = np.array(unsafe_local_xs, dtype=object)
            
            unsafe_h_obj_features = np.zeros((len(unsafe_local_xs), len(unsafe_local_xs[0]), 4), dtype=np.float32)
            for i in range(len(unsafe_local_xs)):
                for j in range(len(unsafe_local_xs[0])):
                    unsafe_h_obj_features[i, j] = unsafe_local_xs[i][j][:4].astype(np.float32)
            training_features["unsafe_h_obj_xs"] = unsafe_h_obj_features
        
        if "unsafe_ee_xs" in data_dict and len(data_dict["unsafe_ee_xs"]) > 0:
            unsafe_ee_xs = data_dict["unsafe_ee_xs"]
            if isinstance(unsafe_ee_xs, list):
                unsafe_ee_xs = np.array(unsafe_ee_xs, dtype=np.float32)
            training_features["unsafe_ee_xs"] = unsafe_ee_xs
        
        return training_features
    
    def normalize_state_sequence(self, local_xs: np.ndarray, normalizer: Any, 
                                 update: bool = False) -> np.ndarray:
        """Normalize state sequence"""
        flattened_local_xs = copy.deepcopy(local_xs.reshape([-1, local_xs.shape[-1]]))
        if update:
            normalizer.experience(flattened_local_xs)
        normalized = normalizer(flattened_local_xs)
        return normalized.reshape(local_xs.shape)
    
    def normalize_training_data(self, features: Dict, h_obj_normalizer: Any, 
                               ee_normalizer: Any, update: bool = False) -> List:
        """Normalize training data"""
        normalized_data = []
        
        if not all(k in features for k in ["safe_h_obj_xs", "safe_ee_xs", 
                                           "unsafe_h_obj_xs", "unsafe_ee_xs"]):
            print("Warning: missing necessary training features")
            return normalized_data
        
        safe_h_obj_xs = features["safe_h_obj_xs"]
        safe_ee_xs = features["safe_ee_xs"]
        unsafe_h_obj_xs = features["unsafe_h_obj_xs"]
        unsafe_ee_xs = features["unsafe_ee_xs"]
        
        safe_h_obj_nxs = features.get("safe_h_obj_nxs", None)
        safe_ee_nxs = features.get("safe_ee_nxs", None)
        
        print(f"Safe feature shape: safe_h_obj_xs={safe_h_obj_xs.shape}, safe_ee_xs={safe_ee_xs.shape}")
        print(f"Unsafe feature shape: unsafe_h_obj_xs={unsafe_h_obj_xs.shape}, unsafe_ee_xs={unsafe_ee_xs.shape}")
        
        # Normalize object history state
        normalized_safe_h_obj_xs = self.normalize_state_sequence(
            safe_h_obj_xs, h_obj_normalizer, update=update).astype(np.float32)
        
        normalized_unsafe_h_obj_xs = self.normalize_state_sequence(
            unsafe_h_obj_xs, h_obj_normalizer, update=False).astype(np.float32)
        
        # Update EE normalizer statistics
        if update:
            ee_normalizer.experience(safe_ee_xs)
            ee_normalizer.experience(unsafe_ee_xs)
        
        # Normalize EE state
        normalized_safe_ee_xs = ee_normalizer(safe_ee_xs).astype(np.float32)
        normalized_unsafe_ee_xs = ee_normalizer(unsafe_ee_xs).astype(np.float32)
        
        # Process optional features
        normalized_safe_h_obj_nxs = None
        normalized_safe_ee_nxs = None
        
        if safe_h_obj_nxs is not None:
            normalized_safe_h_obj_nxs = self.normalize_state_sequence(
                safe_h_obj_nxs, h_obj_normalizer, update=False).astype(np.float32)
        
        if safe_ee_nxs is not None:
            normalized_safe_ee_nxs = ee_normalizer(safe_ee_nxs).astype(np.float32)
        
        # Assemble training data
        normalized_data = [
            normalized_safe_h_obj_xs,
            normalized_safe_h_obj_nxs,
            normalized_safe_ee_xs,
            normalized_safe_ee_nxs,
            normalized_unsafe_h_obj_xs,
            normalized_unsafe_ee_xs
        ]
        
        # Filter out None values
        normalized_data = [data for data in normalized_data if data is not None]
        
        return normalized_data


class SafetyChecker:
    """Manages safety checking and violation detection"""
    
    def __init__(self, env_state: EnvironmentState):
        self.env_state = env_state
        self.env_manager = EnvironmentManager()
        self.geometry_calc = GeometryCalculator()
    
    def check_safety_states(self, env) -> List[Dict]:
        """Check safety states and record safe transitions"""
        safe_states = []
        
        object_states, object_names = self.env_manager.get_object_states_batch(env)
        
        for env_idx in range(env.num_envs):
            env_safe_states = []
            
            for obj_idx, name in enumerate(object_names):
                obj_quat = object_states[env_idx, obj_idx, 3:7]
                tilt_angle_rad = self.geometry_calc.calculate_tilt_angle_from_quaternion_batch(
                    obj_quat.unsqueeze(0)).item()
                tilt_angle_deg = np.degrees(tilt_angle_rad)
                
                if tilt_angle_deg < 15.0:
                    mapping = self.env_state.get_object_mapping(env_idx, name)
                    orig_index = mapping["orig_index"] if mapping else None
                    sample_idx = mapping["sample_idx"] if mapping else None
                    
                    safe_state = {
                        "object_name": name,
                        "tilt_angle_deg": float(f"{tilt_angle_deg:.2f}"),
                        "env_idx": env_idx,
                        "orig_index": orig_index,
                        "sample_idx": sample_idx
                    }
                    env_safe_states.append(safe_state)
                    print(f"Object '{name}' tilt angle {tilt_angle_deg:.2f}° < 15° threshold, now safe!")
            
            if env_safe_states:
                safe_states.append({
                    "env_idx": env_idx,
                    "safe_states": env_safe_states
                })
        
        self.env_state.safe_transitions.extend(safe_states)
        return safe_states
    
    def find_samples_to_relabel(self, safe_state_info: List[Dict], 
                                indices: Dict) -> Tuple[List[int], List[Dict]]:
        """Find samples that need to be relabeled"""
        positions_to_move = []
        mapping_details = []
        
        for safe_state in safe_state_info:
            env_idx = safe_state["env_idx"]
            
            for obj_safe_state in safe_state["safe_states"]:
                obj_name = obj_safe_state["object_name"]
                mapping = self.env_state.get_object_mapping(env_idx, obj_name)
                
                if mapping:
                    orig_index = mapping["orig_index"]
                    
                    for i, (idx, name) in enumerate(zip(indices["unsafe_indices"], 
                                                        indices["unsafe_obj_names"])):
                        if idx == orig_index and name == obj_name:
                            positions_to_move.append(i)
                            mapping_details.append({
                                "position": i,
                                "orig_index": int(orig_index),
                                "object_name": obj_name,
                                "tilt_angle": obj_safe_state["tilt_angle_deg"]
                            })
                            print(f"Found safe sample: position={i}, index={orig_index}, "
                                  f"object={obj_name}, tilt={obj_safe_state['tilt_angle_deg']}°")
        
        return list(set(positions_to_move)), mapping_details


class ModelTrainer:
    """Manages model training"""
    
    def __init__(self, config: Config, training_state: TrainingState, data_manager: DataManager):
        self.config = config
        self.training_state = training_state
        self.data_manager = data_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_performance(self, normalized_data: List) -> float:
        """Evaluate model performance"""
        self.training_state.model.to(self.device)
        self.training_state.model.eval()
        
        # Safe sample prediction
        normalized_safe_h_obj_xs = torch.tensor(normalized_data[0], dtype=torch.float32, device=self.device)
        normalized_safe_ee_xs = torch.tensor(normalized_data[2], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            safe_preds = self.training_state.model([normalized_safe_h_obj_xs, normalized_safe_ee_xs]).cpu().numpy()
        
        # Unsafe sample prediction
        normalized_unsafe_h_obj_xs = torch.tensor(normalized_data[4], dtype=torch.float32, device=self.device)
        normalized_unsafe_ee_xs = torch.tensor(normalized_data[5], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            unsafe_preds = self.training_state.model([normalized_unsafe_h_obj_xs, normalized_unsafe_ee_xs]).cpu().numpy()
        
        # Calculate accuracy
        safe_accuracy = np.mean(safe_preds < 0)
        unsafe_accuracy = np.mean(unsafe_preds > 0)
        performance = (safe_accuracy + unsafe_accuracy) / 2.0
        
        print(f"Model performance evaluation:")
        print(f"  Safe sample accuracy: {safe_accuracy:.4f} ({np.sum(safe_preds < 0)}/{len(safe_preds)})")
        print(f"  Unsafe sample accuracy: {unsafe_accuracy:.4f} ({np.sum(unsafe_preds > 0)}/{len(unsafe_preds)})")
        print(f"  Overall performance: {performance:.4f}")
        
        return performance
    
    def train_batch(self, batch_index: int, training_features: Dict) -> Optional[float]:
        """Train model on batch data"""
        if not all(k in training_features for k in ["safe_h_obj_xs", "safe_ee_xs", 
                                                     "unsafe_h_obj_xs", "unsafe_ee_xs"]):
            print("Warning: missing necessary training features")
            return None
        
        if len(training_features["safe_h_obj_xs"]) < 10 or len(training_features["unsafe_h_obj_xs"]) < 10:
            print(f"Warning: insufficient training samples (safe: {len(training_features.get('safe_h_obj_xs', []))}, "
                  f"unsafe: {len(training_features.get('unsafe_h_obj_xs', []))})")
            return None
        
        # Normalize data
        normalized_data = self.data_manager.normalize_training_data(
            training_features, 
            self.training_state.h_obj_normalizer, 
            self.training_state.ee_normalizer
        )
        
        if not normalized_data:
            print("Warning: normalized data is empty")
            return None
        
        print(f"Safe samples: {len(training_features['safe_h_obj_xs'])}")
        print(f"Unsafe samples: {len(training_features['unsafe_h_obj_xs'])}")
        
        # Save model state before training
        model_state_before = self.training_state.model.state_dict()
        
        # Train model
        self.training_state.model.fit(
            data=normalized_data,
            epoch=self.config.training_epochs,
            verbose_num=1,
            lr=self.config.learning_rate,
            margin_threshold=self.config.margin_threshold,
            derivative_threshold=self.config.derivative_threshold,
            batch_size=self.config.training_batch_size,
            device=self.device
        )
        
        # Evaluate performance
        performance = self.evaluate_performance(normalized_data)
        
        # Record performance
        self.training_state.record_performance(
            batch_index, 
            performance,
            len(training_features["safe_h_obj_xs"]),
            len(training_features["unsafe_h_obj_xs"])
        )
        
        # Save performance history
        performance_history_path = os.path.join(self.config.output_dir, "performance_history.json")
        with open(performance_history_path, 'w') as f:
            json.dump(self.training_state.performance_history, f, indent=2)
        
        # Save best model
        if self.training_state.save_best_model(performance, self.config.output_dir, 
                                              self.config.best_model_path):
            print(f"Batch {batch_index + 1}: new best model saved")
        else:
            print(f"Batch {batch_index + 1}: performance not improved "
                  f"(current: {performance:.4f}, best: {self.training_state.best_performance:.4f})")
            
            # Restore previous model if performance drops significantly
            if performance < self.training_state.best_performance - 0.1:
                print("Performance significantly decreased, restoring previous model state")
                self.training_state.model.load_state_dict(model_state_before)
        
        # Save current batch model
        batch_model_path = os.path.join(self.config.output_dir, f"model_batch_{batch_index + 1}.pt")
        self.training_state.model.save(batch_model_path)
        
        return performance


class ActionPlanner:
    """Plans safe actions using NCBF"""
    
    def __init__(self, training_state: TrainingState, env_state: EnvironmentState, 
                 config: Config):
        self.training_state = training_state
        self.env_state = env_state
        self.config = config
        self.robot_manager = RobotManager()
    
    def update_object_history_buffer(self, env, object_states, object_names):
        """Update object history buffer"""
        for env_idx in range(env.num_envs):
            if env_idx not in self.env_state.object_history_buffer:
                self.env_state.object_history_buffer[env_idx] = {}
            
            for obj_idx, obj_name in enumerate(object_names):
                if obj_name not in self.env_state.object_history_buffer[env_idx]:
                    self.env_state.object_history_buffer[env_idx][obj_name] = []
                
                obj_pos = object_states[env_idx, obj_idx, :3].cpu().numpy()
                obj_quat = object_states[env_idx, obj_idx, 3:7]
                tilt_angle_rad = GeometryCalculator().calculate_tilt_angle_from_quaternion_batch(
                    obj_quat.unsqueeze(0)).item()
                
                if not self.env_state.object_history_buffer[env_idx][obj_name]:
                    base_pos = obj_pos
                else:
                    first_state = self.env_state.object_history_buffer[env_idx][obj_name][0]
                    base_pos = obj_pos.copy()
                    base_pos[0] -= first_state[1]
                    base_pos[1] -= first_state[2]
                    base_pos[2] -= first_state[3]
                
                rel_pos = obj_pos - base_pos
                state = [tilt_angle_rad, rel_pos[0], rel_pos[1], rel_pos[2]]
                
                self.env_state.object_history_buffer[env_idx][obj_name].append(state)
                
                if len(self.env_state.object_history_buffer[env_idx][obj_name]) > self.config.horizon:
                    self.env_state.object_history_buffer[env_idx][obj_name] = \
                        self.env_state.object_history_buffer[env_idx][obj_name][-self.config.horizon:]
    
    def move_ee_towards_object_batch(self, env, ee_positions, obj_positions,
                                    env_indices, obj_indices, object_states,
                                    object_names, move_step_size=0.01,
                                    grid_size=8, safety_threshold=-0.05):
        """Batch compute safe actions for multiple environments"""
        num_envs_to_process = len(env_indices)
        device = env.device
        
        # Generate action grid
        x_moves = np.linspace(-move_step_size, move_step_size, grid_size)
        y_moves = np.linspace(-move_step_size, move_step_size, grid_size)
        action_grid = np.array(list(itertools.product(x_moves, y_moves)))
        num_actions = len(action_grid)
        
        # Collect batch object histories
        batch_obj_histories = []
        batch_obj_positions = []
        
        for i, env_idx in enumerate(env_indices):
            obj_idx = obj_indices[i]
            obj_name = object_names[obj_idx]
            
            obj_history = []
            if env_idx in self.env_state.object_history_buffer and \
               obj_name in self.env_state.object_history_buffer[env_idx]:
                obj_history = np.array(self.env_state.object_history_buffer[env_idx][obj_name], 
                                      dtype=np.float32)
            
            if len(obj_history) < self.config.horizon:
                obj_pos = object_states[env_idx, obj_idx, :3].cpu().numpy()
                obj_quat = object_states[env_idx, obj_idx, 3:7]
                tilt_angle_rad = GeometryCalculator().calculate_tilt_angle_from_quaternion_batch(
                    obj_quat.unsqueeze(0)).item()
                current_state = [tilt_angle_rad, 0.0, 0.0, 0.0]
                obj_history = np.array([current_state for _ in range(self.config.horizon)], 
                                      dtype=np.float32)
            
            if len(obj_history) > self.config.horizon:
                obj_history = obj_history[-self.config.horizon:]
            
            batch_obj_histories.append(obj_history)
            batch_obj_positions.append(object_states[env_idx, obj_idx, :3].cpu().numpy()[:2])
        
        batch_obj_histories = np.stack(batch_obj_histories)
        batch_obj_positions = np.stack(batch_obj_positions)
        batch_ee_positions = ee_positions.cpu().numpy()[:, :2]
        
        # Expand for all actions
        expanded_histories = np.repeat(batch_obj_histories[:, np.newaxis], num_actions, axis=1)
        predicted_positions = batch_ee_positions[:, np.newaxis, :] + action_grid[np.newaxis, :, :]
        rel_positions = predicted_positions - batch_obj_positions[:, np.newaxis, :]
        
        # Normalize
        flat_histories = expanded_histories.reshape(-1, self.config.horizon, 4)
        normalized_histories = DataManager(self.config).normalize_state_sequence(
            flat_histories, self.training_state.h_obj_normalizer, update=False)
        normalized_histories = normalized_histories.reshape(num_envs_to_process, num_actions, 
                                                           self.config.horizon, 4)
        
        flat_rel_positions = rel_positions.reshape(-1, 2)
        normalized_rel_positions = self.training_state.ee_normalizer(flat_rel_positions)
        normalized_rel_positions = normalized_rel_positions.reshape(num_envs_to_process, 
                                                                    num_actions, 2)
        
        # Convert to tensors
        hist_tensor = torch.tensor(normalized_histories, dtype=torch.float32, device=device)
        rel_pos_tensor = torch.tensor(normalized_rel_positions, dtype=torch.float32, device=device)
        
        # Flatten for inference
        hist_flat = hist_tensor.reshape(-1, self.config.horizon, 4)
        rel_pos_flat = rel_pos_tensor.reshape(-1, 2)
        
        # Single model inference
        with torch.no_grad():
            cbf_values = self.training_state.model([hist_flat, rel_pos_flat])
        
        cbf_values = cbf_values.reshape(num_envs_to_process, num_actions).cpu().numpy()
        
        # Select best action for each environment
        best_actions = []
        distances = []
        
        for i in range(num_envs_to_process):
            env_idx = env_indices[i]
            distance = np.linalg.norm(batch_obj_positions[i] - batch_ee_positions[i])
            distances.append(distance)
            
            if distance > 0.01:
                best_idx = np.argmin(cbf_values[i])
                best_cbf = cbf_values[i, best_idx]
                
                if best_cbf < safety_threshold:
                    best_action = action_grid[best_idx]
                    best_actions.append(best_action)
                    
                    if env_idx % 10 == 0:
                        print(f"Env {env_idx} - safe action: [{best_action[0]:.4f}, "
                              f"{best_action[1]:.4f}], CBF: {best_cbf:.4f}")
                else:
                    best_actions.append(np.array([0.0, 0.0]))
                    print(f"Env {env_idx} - all actions unsafe (best CBF: {best_cbf:.4f}), stay still")
            else:
                best_actions.append(np.array([0.0, 0.0]))
        
        return np.array(best_actions), np.array(distances)


class EnvironmentController:
    """Controls environment operations"""
    
    def __init__(self, config: Config, batch_state: BatchState, env_state: EnvironmentState):
        self.config = config
        self.batch_state = batch_state
        self.env_state = env_state
        self.robot_manager = RobotManager()
        self.env_manager = EnvironmentManager()
    
    def clear_scene(self):
        """Clear existing elements in scene"""
        import omni.usd
        from pxr import Usd
        
        stage = omni.usd.get_context().get_stage()
        env_path = "/World/envs"
        
        if stage.GetPrimAtPath(env_path).IsValid():
            print(f"Clearing environment: {env_path}")
            stage.RemovePrim(env_path)
        
        print("Scene cleared")
    
    def restore_state_from_json(self, env, env_ids=None):
        """Restore state from boundary samples"""
        current_batch_samples = self.batch_state.get_current_batch_samples()
        
        print(f"\n===== Restore batch {self.batch_state.current_index + 1}/"
              f"{self.batch_state.total_batches} =====")
        print(f"Processing {len(current_batch_samples)} samples")
        
        # Convert env_ids to list if needed
        if env_ids is None:
            env_ids = list(range(env.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.cpu().tolist()
        elif isinstance(env_ids, slice):
            env_ids = list(range(env.num_envs))[env_ids]
        
        # Prepare environment data
        env_data = []
        for i, env_id in enumerate(env_ids):
            sample_idx = i % len(current_batch_samples)
            env_state = current_batch_samples[sample_idx]
            
            orig_index = env_state.get("orig_index", sample_idx)
            self.env_state.snapshot_time_indices[env_id] = orig_index
            
            # Record object information
            for obj_info in env_state["objects_state"]:
                obj_name = obj_info["name"]
                self.env_state.update_object_mapping(env_id, obj_name, {
                    "orig_index": orig_index,
                    "sample_idx": sample_idx,
                    "object_data": obj_info
                })
            
            env_data.append({
                "env_id": env_id,
                "sample_idx": sample_idx,
                "robot_joints": env_state["robot_joints"],
                "objects_state": env_state["objects_state"],
                "ee_state": env_state.get("ee_state", None)
            })
        
        # Apply robot joint state
        robot = env.scene.articulations.get("robot", None)
        env_origins = env.scene.env_origins
        
        print("\n===== Restore robot state =====")
        for data in env_data:
            env_id = data["env_id"]
            env_tensor_id = torch.tensor([env_id], device=env.device)
            
            joint_pos = torch.tensor(data["robot_joints"], device=env.device).unsqueeze(0)
            joint_vel = torch.zeros_like(joint_pos)
            
            robot.set_joint_position_target(joint_pos, env_ids=env_tensor_id)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_tensor_id)
            print(f"Environment {env_id}: robot joint state restored")
        
        # Render to update robot state
        for _ in range(5):
            env.sim.render()
            time.sleep(0.02)
        
        # Get actual EE positions
        actual_ee_positions = self.robot_manager.get_ee_positions_batch(env)
        
        # Restore object states
        for data in env_data:
            env_id = data["env_id"]
            env_tensor_id = torch.tensor([env_id], device=env.device)
            
            position_offset = torch.zeros(3, device=env.device)
            
            if data["ee_state"] is not None and actual_ee_positions is not None:
                json_ee_pos = torch.tensor([
                    data["ee_state"]["x"],
                    data["ee_state"]["y"],
                    data["ee_state"]["z"]
                ], device=env.device)
                
                actual_ee_pos = actual_ee_positions[env_id]
                position_offset = actual_ee_pos - json_ee_pos
                print(f"Environment {env_id} - EE position offset: {position_offset.cpu().numpy()}")
            
            for obj_info in data["objects_state"]:
                original_name = obj_info["name"]
                obj = None
                
                if hasattr(env.scene, "rigid_objects"):
                    obj = env.scene.rigid_objects[original_name]
                
                if obj is None:
                    print(f"Warning: cannot find object '{original_name}'")
                    continue
                
                world_position = torch.tensor([
                    obj_info["position"]["x"],
                    obj_info["position"]["y"],
                    obj_info["position"]["z"]
                ], device=env.device).unsqueeze(0)
                
                local_position = world_position + position_offset.unsqueeze(0)
                
                orientation = torch.tensor([
                    obj_info["orientation"]["qx"],
                    obj_info["orientation"]["qy"],
                    obj_info["orientation"]["qz"],
                    obj_info["orientation"]["qw"]
                ], device=env.device).unsqueeze(0)
                
                lin_vel = torch.zeros_like(local_position)
                ang_vel = torch.zeros_like(local_position)
                
                root_state = torch.cat([local_position, orientation, lin_vel, ang_vel], dim=1)
                obj.write_root_state_to_sim(root_state, env_tensor_id)
                
                # Restore mass
                if "mass" in obj_info and hasattr(obj, "root_physx_view"):
                    try:
                        current_masses = obj.root_physx_view.get_masses()
                        masses = torch.clone(current_masses)
                        masses[env_id] = float(obj_info["mass"])
                        masses = masses.cpu()
                        cpu_env_tensor_id = env_tensor_id.cpu()
                        obj.root_physx_view.set_masses(masses, cpu_env_tensor_id)
                    except Exception as e:
                        print(f"Warning: failed to set mass for '{original_name}': {e}")
                
                # Restore material properties
                if "material" in obj_info and hasattr(obj, "root_physx_view"):
                    try:
                        material_data = obj_info["material"]
                        current_materials = obj.root_physx_view.get_material_properties()
                        materials = torch.clone(current_materials)
                        
                        static_friction = float(material_data["static_friction"])
                        dynamic_friction = float(material_data["dynamic_friction"])
                        restitution = float(material_data["restitution"])
                        
                        num_shapes = materials.size(1)
                        for shape_idx in range(num_shapes):
                            materials[env_id, shape_idx, 0] = static_friction
                            materials[env_id, shape_idx, 1] = dynamic_friction
                            materials[env_id, shape_idx, 2] = restitution
                        
                        materials = materials.cpu()
                        cpu_env_tensor_id = env_tensor_id.cpu()
                        obj.root_physx_view.set_material_properties(materials, cpu_env_tensor_id)
                    except Exception as e:
                        print(f"Warning: failed to set material for '{original_name}': {e}")
                
                # Restore COM
                if "com" in obj_info and hasattr(obj, "root_physx_view"):
                    try:
                        current_coms = obj.root_physx_view.get_coms()
                        coms = torch.clone(current_coms)
                        
                        com_data = obj_info["com"]
                        coms[env_id, 0] = float(com_data.get("x", 0.0))
                        coms[env_id, 1] = float(com_data.get("y", 0.0))
                        coms[env_id, 2] = float(com_data.get("z", 0.0))
                        
                        coms = coms.cpu()
                        cpu_env_tensor_id = env_tensor_id.cpu()
                        obj.root_physx_view.set_coms(coms, cpu_env_tensor_id)
                    except Exception as e:
                        print(f"Warning: failed to set COM for '{original_name}': {e}")
                
                print(f"Environment {env_id} - object '{original_name}' state restored")
        
        # Render to apply states
        for _ in range(5):
            env.sim.render()
            time.sleep(0.02)
        
        print(f"Batch {self.batch_state.current_index + 1}/{self.batch_state.total_batches} restored!")
    
    def create_environment(self):
        """Create environment with state restoration"""
        env_cfg = parse_env_cfg(self.config.task, device=self.config.device, 
                                num_envs=self.config.num_envs)
        env_cfg.env_name = self.config.task
        
        if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'num_envs'):
            env_cfg.scene.num_envs = self.config.num_envs
        
        if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'num_envs'):
            env_cfg.sim.num_envs = self.config.num_envs
        
        env_cfg.terminations.time_out = None
        
        if "Lift" in self.config.task:
            env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
            env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
        
        if not hasattr(env_cfg, 'events'):
            env_cfg.events = type('EventsCfg', (), {})()
        
        # Register event with correct signature: func(env, env_ids=None)
        def restore_wrapper(env, env_ids=None):
            return self.restore_state_from_json(env, env_ids)
        
        env_cfg.events.restore_state = EventTerm(
            func=restore_wrapper,
            mode="reset",
            params={}
        )
        
        env = gym.make(self.config.task, cfg=env_cfg).unwrapped
        
        env.reset()
        
        for _ in range(5):
            env.sim.render()
            time.sleep(0.02)
        
        return env
    
    def proceed_to_next_batch(self, env) -> bool:
        """Proceed to next batch"""
        if self.batch_state.is_complete():
            print("\nAll batches processed!")
            return False
        
        print(f"\n====== Prepare next batch {self.batch_state.current_index + 1}/"
              f"{self.batch_state.total_batches} ======")
        
        env.reset()
        
        for _ in range(5):
            env.sim.render()
            time.sleep(0.02)
        
        return True


class BatchProcessor:
    """Processes batches and coordinates training"""
    
    def __init__(self, config: Config, training_state: TrainingState, 
                 data_state: DataState, batch_state: BatchState,
                 env_state: EnvironmentState, training_buffer: TrainingBuffer):
        self.config = config
        self.training_state = training_state
        self.data_state = data_state
        self.batch_state = batch_state
        self.env_state = env_state
        self.training_buffer = training_buffer
        
        self.data_manager = DataManager(config)
        self.safety_checker = SafetyChecker(env_state)
        self.model_trainer = ModelTrainer(config, training_state, self.data_manager)
        self.action_planner = ActionPlanner(training_state, env_state, config)
        self.robot_manager = RobotManager()
        self.env_manager = EnvironmentManager()
    
    def process_batch(self, env):
        """Process current batch"""
        print(f"\n====== Batch {self.batch_state.current_index + 1}/"
              f"{self.batch_state.total_batches} safety status ======")
        
        current_batch_orig_indices = self.batch_state.get_current_batch_indices()
        
        # Print EE positions
        ee_positions = self.robot_manager.get_ee_positions_batch(env)
        if ee_positions is not None:
            print("\nInitial end-effector positions:")
            for env_idx in range(min(5, env.num_envs)):
                pos = ee_positions[env_idx].cpu().numpy()
                print(f"Env {env_idx} - position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
        
        print("\n====== Start NCBF safety movement test ======")
        num_move_steps = 2
        move_step_size = 0.01
        grid_size = 8
        safety_threshold = 0.05
        
        active_envs = list(range(env.num_envs))
        
        # Initialize object history buffer
        object_states, object_names = self.env_manager.get_object_states_batch(env)
        
        print("Initializing object history buffer...")
        for _ in range(self.config.horizon):
            self.action_planner.update_object_history_buffer(env, object_states, object_names)
        
        # Movement loop
        for step in range(num_move_steps):
            print(f"\nStep {step+1}/{num_move_steps}")
            
            object_states, object_names = self.env_manager.get_object_states_batch(env)
            distances_tensor, object_names_list, closest_info = \
                self.robot_manager.calculate_distances_to_objects_batch(env)
            
            # Collect environments to process
            env_indices_to_process = []
            obj_indices_to_process = []
            ee_positions_to_process = []
            obj_positions_to_process = []
            
            for env_idx in active_envs:
                if env_idx in closest_info:
                    closest = closest_info[env_idx]
                    closest_name = closest['name']
                    
                    if closest_name in object_names_list:
                        obj_idx = object_names_list.index(closest_name)
                        
                        env_indices_to_process.append(env_idx)
                        obj_indices_to_process.append(obj_idx)
                        ee_positions_to_process.append(ee_positions[env_idx])
                        obj_positions_to_process.append(object_states[env_idx, obj_idx, :3])
            
            # Batch compute actions
            if env_indices_to_process:
                batch_ee_positions = torch.stack(ee_positions_to_process)
                batch_obj_positions = torch.stack(obj_positions_to_process)
                
                batch_actions_2d, batch_distances = self.action_planner.move_ee_towards_object_batch(
                    env, batch_ee_positions, batch_obj_positions,
                    env_indices_to_process, obj_indices_to_process,
                    object_states, object_names,
                    move_step_size=move_step_size,
                    grid_size=grid_size,
                    safety_threshold=safety_threshold
                )
                
                actions = torch.zeros((env.num_envs, 3), device=env.device)
                for i, env_idx in enumerate(env_indices_to_process):
                    actions[env_idx, 0] = batch_actions_2d[i, 0]
                    actions[env_idx, 1] = batch_actions_2d[i, 1]
                    actions[env_idx, 2] = -1.0
                    
                    if step == 0 or step == num_move_steps - 1 or env_idx % 20 == 0:
                        print(f"Env {env_idx} - distance: {batch_distances[i]:.4f}")
            else:
                actions = torch.zeros((env.num_envs, 3), device=env.device)
            
            # Execute action
            env.step(actions)
            
            # Update states
            ee_positions = self.robot_manager.get_ee_positions_batch(env)
            object_states, object_names = self.env_manager.get_object_states_batch(env)
            
            self.action_planner.update_object_history_buffer(env, object_states, object_names)
            
            env.sim.render()
            time.sleep(0.05)
        
        print("End-effector movement test completed")
        
        # Check safety
        safe_states = self.safety_checker.check_safety_states(env)
        
        # Record batch results
        self.batch_state.results[self.batch_state.current_index] = {
            "batch_index": self.batch_state.current_index,
            "safe_states": safe_states,
            "has_safe_transition": len(safe_states) > 0
        }
        
        # Process safe transitions
        positions_to_move = []
        mapping_details = []
        
        if safe_states:
            safe_states_file = os.path.join(self.config.output_dir, 
                                           f"batch_{self.batch_state.current_index+1}_safe_states.json")
            with open(safe_states_file, 'w') as f:
                json.dump(safe_states, f, indent=2)
            
            print(f"\nBatch {self.batch_state.current_index+1}: found {len(safe_states)} "
                  f"environments with safe transitions")
            
            # Update safety labels
            positions_to_move, mapping_details = self.safety_checker.find_samples_to_relabel(
                safe_states, self.data_state.get_indices_dict())
            
            if positions_to_move:
                self.data_state.update_from_relabeling(positions_to_move)
                
                batch_mapping_path = os.path.join(
                    self.config.output_dir, 
                    f"relabeled_samples_batch_{self.batch_state.current_index+1}.json"
                )
                with open(batch_mapping_path, 'w') as f:
                    json.dump(mapping_details, f, indent=2)
                
                print(f"Batch {self.batch_state.current_index+1}: relabeled "
                      f"{len(positions_to_move)} samples")
        
        # Training
        print(f"\n===== Train model with batch {self.batch_state.current_index+1} data =====")
        
        if self.config.use_buffer:
            # Extract all current batch data
            batch_data_dict, batch_indices_dict = self.extract_current_batch_data(
                extract_all=True, batch_indices=current_batch_orig_indices)
            
            # Add to buffer
            self.training_buffer.add_batch(batch_data_dict, batch_indices_dict)
            
            # Train with buffer data
            if self.training_buffer.current_size > 0:
                training_features = self.data_manager.extract_training_features(
                    self.training_buffer.to_dict())
                print(f"Using buffer data: {self.training_buffer.current_size}/"
                      f"{self.training_buffer.max_size} batches")
            else:
                print("Buffer empty, skip training")
                self.batch_state.next_batch()
                return
        else:
            # Use global data
            training_features = self.data_manager.extract_training_features(
                self.data_state.to_dict())
            print(f"Using global data: {len(self.data_state.safe_local_xs)} safe, "
                  f"{len(self.data_state.unsafe_local_xs)} unsafe")
        
        # Train
        if training_features:
            self.model_trainer.train_batch(self.batch_state.current_index, training_features)
        
        self.batch_state.next_batch()
    
    def extract_current_batch_data(self, extract_all=False, batch_indices=None):
        """Extract current batch data"""
        batch_data = {}
        batch_indices_out = {}
        
        if extract_all and batch_indices is not None:
            # Initialize
            batch_data["safe_local_xs"] = []
            batch_data["safe_local_nxs"] = []
            batch_data["safe_ee_xs"] = []
            batch_data["safe_ee_nxs"] = []
            batch_data["unsafe_local_xs"] = []
            batch_data["unsafe_ee_xs"] = []
            batch_indices_out["safe_indices"] = []
            batch_indices_out["unsafe_indices"] = []
            batch_indices_out["safe_obj_names"] = []
            batch_indices_out["unsafe_obj_names"] = []
            
            # Extract safe samples
            for i, idx in enumerate(self.data_state.safe_indices):
                if idx in batch_indices:
                    batch_data["safe_local_xs"].append(self.data_state.safe_local_xs[i])
                    if len(self.data_state.safe_local_nxs) > i:
                        batch_data["safe_local_nxs"].append(self.data_state.safe_local_nxs[i])
                    if len(self.data_state.safe_ee_xs) > i:
                        batch_data["safe_ee_xs"].append(self.data_state.safe_ee_xs[i])
                    if len(self.data_state.safe_ee_nxs) > i:
                        batch_data["safe_ee_nxs"].append(self.data_state.safe_ee_nxs[i])
                    batch_indices_out["safe_indices"].append(idx)
                    batch_indices_out["safe_obj_names"].append(self.data_state.safe_obj_names[i])
            
            # Extract unsafe samples
            for i, idx in enumerate(self.data_state.unsafe_indices):
                if idx in batch_indices:
                    batch_data["unsafe_local_xs"].append(self.data_state.unsafe_local_xs[i])
                    if len(self.data_state.unsafe_ee_xs) > i:
                        batch_data["unsafe_ee_xs"].append(self.data_state.unsafe_ee_xs[i])
                    batch_indices_out["unsafe_indices"].append(idx)
                    batch_indices_out["unsafe_obj_names"].append(self.data_state.unsafe_obj_names[i])
            
            # Convert to arrays
            for key in batch_data:
                if len(batch_data[key]) > 0:
                    if "xs" in key:
                        batch_data[key] = np.array(batch_data[key], 
                                                   dtype=object if "local" in key else np.float32)
            
            for key in batch_indices_out:
                if len(batch_indices_out[key]) > 0:
                    batch_indices_out[key] = np.array(batch_indices_out[key])
        
        return batch_data, batch_indices_out


# ============================================================================
# Main Application
# ============================================================================

class Application:
    """Main application orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize state objects
        self.training_state = None
        self.data_state = DataState()
        self.batch_state = BatchState()
        self.env_state = EnvironmentState()
        self.training_buffer = None
        
        # Initialize managers
        self.data_manager = DataManager(config)
        self.env_controller = None
        self.batch_processor = None
        
        self.env = None
    
    def initialize(self):
        """Initialize application components"""
        print("====== Initialize application ======")
        
        # Load model and normalizers
        self.training_state = TrainingState.from_paths(
            self.config.model_path,
            self.config.h_obj_normalizer_path,
            self.config.ee_normalizer_path
        )
        
        # Load and process data
        json_data = self.data_manager.load_json_data(self.config.json_file)
        
        print("\n====== Calculate CBF values and select boundary samples ======")
        organized = self.data_manager.organize_by_object(json_data, max_tilt_angle=91)
        print(f"Organized {len(organized)} objects")
        
        self.data_state, indices = self.data_manager.build_sequences(
            organized, self.config.horizon, self.config.step, self.config.min_seq_len
        )
        
        print(f"Generated {len(self.data_state.safe_local_xs)} safe samples")
        
        # Calculate CBF values
        features_data = self.prepare_features_data()
        all_cbf_values = self.calculate_cbf_values(features_data)
        
        # Extract boundary samples
        boundary_info = self.extract_boundary_samples(features_data, indices, all_cbf_values)
        print(f"Found {len(boundary_info)} samples near safety boundary")
        
        # Prepare boundary samples
        boundary_samples_data = []
        for sample in boundary_info:
            orig_index = sample["orig_index"]
            if orig_index < len(json_data):
                boundary_samples_data.append(json_data[orig_index])
                boundary_samples_data[-1]["cbf_value"] = sample["cbf_value"]
                boundary_samples_data[-1]["orig_index"] = orig_index
        
        # Initialize batch state
        self.batch_state.boundary_samples = boundary_samples_data
        self.batch_state.batch_size = self.config.num_envs
        self.batch_state.total_batches = (len(boundary_samples_data) + 
                                         self.batch_state.batch_size - 1) // self.batch_state.batch_size
        
        print(f"\nExtracted {len(boundary_samples_data)} boundary samples, "
              f"{self.batch_state.total_batches} batches")
        
        # Save boundary indices
        boundary_indices_path = os.path.join(self.config.output_dir, "boundary_indices.json")
        with open(boundary_indices_path, 'w') as f:
            boundary_indices = [sample["orig_index"] for sample in boundary_info]
            json.dump(boundary_indices, f, indent=2)
        
        # Initialize training buffer
        self.training_buffer = TrainingBuffer(max_size=self.config.buffer_size)
        
        # Initialize controllers
        self.env_controller = EnvironmentController(self.config, self.batch_state, self.env_state)
        self.batch_processor = BatchProcessor(
            self.config, self.training_state, self.data_state,
            self.batch_state, self.env_state, self.training_buffer
        )
    
    def prepare_features_data(self):
        """Prepare features for CBF calculation"""
        features_data = {}
        
        if len(self.data_state.unsafe_local_xs) > 0:
            unsafe_local_xs = self.data_state.unsafe_local_xs
            unsafe_features = np.zeros((unsafe_local_xs.shape[0], unsafe_local_xs.shape[1], 4), 
                                      dtype=np.float32)
            
            for i in range(unsafe_local_xs.shape[0]):
                for j in range(unsafe_local_xs.shape[1]):
                    unsafe_features[i, j] = unsafe_local_xs[i, j, :4].astype(np.float32)
            
            features_data["unsafe_local_xs"] = unsafe_features
        
        if self.data_state.unsafe_ee_xs.size > 0:
            features_data["unsafe_ee_xs"] = self.data_state.unsafe_ee_xs.astype(np.float32)
        
        return features_data
    
    def calculate_cbf_values(self, features_data, batch_size=128):
        """Calculate CBF values for unsafe samples"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_state.model.to(device)
        self.training_state.model.eval()
        
        all_cbf_values = {}
        
        if "unsafe_local_xs" in features_data and "unsafe_ee_xs" in features_data:
            unsafe_h_obj_xs = features_data["unsafe_local_xs"]
            unsafe_ee_xs = features_data["unsafe_ee_xs"]
            
            unsafe_cbf_values = []
            
            print(f"Processing {len(unsafe_h_obj_xs)} unsafe samples CBF values...")
            
            num_batches = (len(unsafe_h_obj_xs) + batch_size - 1) // batch_size
            for i in tqdm(range(num_batches), desc="Processing unsafe samples"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(unsafe_h_obj_xs))
                
                batch_h_obj_xs = unsafe_h_obj_xs[start_idx:end_idx]
                batch_ee_xs = unsafe_ee_xs[start_idx:end_idx]
                
                normalized_batch_h_obj_xs = np.zeros_like(batch_h_obj_xs)
                for t in range(batch_h_obj_xs.shape[1]):
                    normalized_batch_h_obj_xs[:, t, :] = self.training_state.h_obj_normalizer(
                        batch_h_obj_xs[:, t, :])
                
                normalized_batch_ee_xs = self.training_state.ee_normalizer(batch_ee_xs)
                
                h_obj_xs_tensor = torch.tensor(normalized_batch_h_obj_xs, 
                                              dtype=torch.float32, device=device)
                ee_xs_tensor = torch.tensor(normalized_batch_ee_xs, 
                                           dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    batch_cbf_values = self.training_state.model([h_obj_xs_tensor, 
                                                                  ee_xs_tensor]).cpu().numpy().flatten()
                
                unsafe_cbf_values.extend(batch_cbf_values)
            
            all_cbf_values["unsafe"] = np.array(unsafe_cbf_values)
        
        return all_cbf_values
    
    def extract_boundary_samples(self, features_data, indices, all_cbf_values):
        """Extract samples near safety boundary"""
        boundary_info = []
        
        if "unsafe" in all_cbf_values and "unsafe_local_xs" in features_data:
            unsafe_cbf_values = all_cbf_values["unsafe"]
            boundary_indices = np.where((unsafe_cbf_values >= self.config.cbf_min) & 
                                       (unsafe_cbf_values <= self.config.cbf_max))[0]
            
            if len(boundary_indices) > 0:
                unsafe_indices = indices["unsafe_indices"]
                unsafe_obj_names = indices["unsafe_obj_names"]
                
                for i, idx in enumerate(boundary_indices):
                    orig_idx = unsafe_indices[idx]
                    obj_name = unsafe_obj_names[idx]
                    
                    sample_info = {
                        "sample_index": int(idx),
                        "orig_index": int(orig_idx),
                        "object_name": str(obj_name),
                        "cbf_value": float(unsafe_cbf_values[idx])
                    }
                    boundary_info.append(sample_info)
        
        return boundary_info
    
    def run(self):
        """Run the main application loop"""
        print("\n====== Start scene reconstruction ======")
        
        # Clear and create environment
        self.env_controller.clear_scene()
        self.env = self.env_controller.create_environment()
        
        # Process first batch
        self.batch_processor.process_batch(self.env)
        
        ACTION_DIM = 3
        
        # Batch processing loop
        try:
            while not self.batch_state.is_complete():
                print(f"\nWait {self.config.batch_interval} seconds to process next batch...")
                
                # Maintain scene rendering
                for _ in range(int(self.config.batch_interval * 10)):
                    self.env.step(torch.zeros((self.env.num_envs, ACTION_DIM), device=self.env.device))
                    self.env.sim.render()
                    time.sleep(0.1)
                
                # Process next batch
                if self.env_controller.proceed_to_next_batch(self.env):
                    self.batch_processor.process_batch(self.env)
                else:
                    break
            
            # Save final results
            self.save_final_results()
            
            print("\n====== Refinement training done, press Ctrl+C to quit ======")
            
            while True:
                self.env.step(torch.zeros((self.env.num_envs, ACTION_DIM), device=self.env.device))
                self.env.sim.render()
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nUser interrupt, quit")
        
        finally:
            if self.env:
                self.env.close()
            print("Environment closed")
    
    def save_final_results(self):
        """Save final results"""
        final_output_path = os.path.join(self.config.output_dir, "final_results.json")
        with open(final_output_path, 'w') as f:
            json.dump(self.batch_state.results, f, indent=2)
        
        print(f"\nAll batches processed, results saved to {final_output_path}")
        
        print("\n===== Save updated dataset =====")
        final_dataset_path = os.path.join(self.config.output_dir, "final_updated_dataset.npz")
        np.savez(
            final_dataset_path,
            safe_local_xs=self.data_state.safe_local_xs,
            safe_local_nxs=self.data_state.safe_local_nxs,
            safe_ee_xs=self.data_state.safe_ee_xs,
            safe_ee_nxs=self.data_state.safe_ee_nxs,
            unsafe_local_xs=self.data_state.unsafe_local_xs,
            unsafe_ee_xs=self.data_state.unsafe_ee_xs,
            safe_indices=self.data_state.safe_indices,
            unsafe_indices=self.data_state.unsafe_indices,
            safe_obj_names=self.data_state.safe_obj_names,
            unsafe_obj_names=self.data_state.unsafe_obj_names
        )
        
        # Save statistics
        stats_path = os.path.join(self.config.output_dir, "final_dataset_stats.json")
        total_relabeled = sum(len(result["safe_states"]) 
                             for result in self.batch_state.results.values() 
                             if result["has_safe_transition"])
        stats = {
            "safe_samples": len(self.data_state.safe_local_xs),
            "unsafe_samples": len(self.data_state.unsafe_local_xs),
            "total_samples": len(self.data_state.safe_local_xs) + len(self.data_state.unsafe_local_xs),
            "relabeled_samples_total": total_relabeled,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Final dataset saved to {final_dataset_path}")
        print(f"Dataset statistics saved to {stats_path}")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    config = Config.from_args(args_cli)
    
    app = Application(config)
    app.initialize()
    app.run()


if __name__ == "__main__":
    main()
    simulation_app.close()