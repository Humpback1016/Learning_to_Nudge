import json
import numpy as np
import argparse
import math
from collections import defaultdict
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Reformat JSON data to NCBF training format")
    parser.add_argument("--input", type=str, default="ncbf_trajectories/trajectories.json", help="input json file path")
    parser.add_argument("--output", type=str, default=None, help="output npz/pickle file path")
    parser.add_argument("--horizon", type=int, default=6, help="history window size")
    parser.add_argument("--step", type=int, default=2, help="next step size")
    parser.add_argument("--min_seq_len", type=int, default=15, help="minimum sequence length")
    parser.add_argument("--max_tilt_angle", type=float, default=30.0, help="allowed max tilt angle")
    return parser.parse_args()

def load_json_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def organize_by_object(json_data, max_tilt_rad):
    # data_entry: {"ee_state": {...}, "objects_state": [...], "is_safe": [[true], ...]}
    organized = defaultdict(lambda: defaultdict(list))
    large_tilt_count = 0  
    total_samples = 0  
    
    for t, entry in enumerate(json_data):
        objects = entry["objects_state"]
        is_safe = entry.get("is_safe", [[True] for _ in objects])
        ee_state = entry["ee_state"]
        
        for obj_idx, (obj, safe_flag) in enumerate(zip(objects, is_safe)):
            total_samples += 1
            # get tilt_cost (radians)
            tilt_angle_rad = obj.get("tilt_cost", 0)
            
            # check if tilt angle is greater than threshold
            if abs(tilt_angle_rad) > max_tilt_rad:
                large_tilt_count += 1
                continue  # skip this sample
                
            # only process the first boolean value
            safe = safe_flag[0] if isinstance(safe_flag, list) else safe_flag
            
            # get object name
            obj_name = obj.get("name", f"object_{obj_idx}")
            
            data_point = {
                "t": t,
                "ee_state": ee_state,
                "obj_state": obj,
                "is_safe": safe,
                "obj_name": obj_name 
            }
            organized[obj_idx][t] = data_point
            
    return organized, large_tilt_count, total_samples

# object position is relative to itself
def build_sequences(organized, horizon, step, min_seq_len):
    safe_local_xs, safe_local_nxs, safe_ee_xs, safe_ee_nxs = [], [], [], []
    unsafe_local_xs, unsafe_ee_xs = [], []
    
    for obj_idx, time_dict in tqdm(organized.items(), desc="Processing objects"):
        # sort by time
        times = sorted(time_dict.keys())
        if len(times) < min_seq_len:
            continue
        
        for i in range(horizon, len(times) - step):
            # current window and next window time points
            idxs = times[i-horizon:i]
            next_idxs = times[i-horizon+1:i+1]
            curr_idx = times[i]
            next_idx = times[i+step]
            
            # select reference base point (start point of sliding window)
            base_idx = idxs[0]
            nbase_idx = next_idxs[0]
            
            # 获取对象名称
            obj_name = time_dict[base_idx]["obj_name"]
            
            # get base point position
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
            
            # build current window object history (relative to base)
            local_x = []
            for j in idxs:
                obj_tilt = time_dict[j]["obj_state"]["tilt_cost"]
                obj_pos = np.array([
                    time_dict[j]["obj_state"]["position"]["x"],
                    time_dict[j]["obj_state"]["position"]["y"],
                    time_dict[j]["obj_state"]["position"]["z"]
                ])
                # convert to relative coordinates to base
                rel_pos = obj_pos - base_pos
          
                features = [obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]]
                features.append(time_dict[j]["t"])  # add time index
                features.append(obj_name)  # add object name
                local_x.append(features)
            
            # build next window object history (relative to nbase)
            local_nx = []
            for j in next_idxs:
                obj_tilt = time_dict[j]["obj_state"]["tilt_cost"]
                obj_pos = np.array([
                    time_dict[j]["obj_state"]["position"]["x"],
                    time_dict[j]["obj_state"]["position"]["y"],
                    time_dict[j]["obj_state"]["position"]["z"]
                ])
                # convert to relative coordinates to nbase
                rel_pos = obj_pos - nbase_pos
                
                features = [obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]]
                features.append(time_dict[j]["t"]) 
                features.append(obj_name)  
                local_nx.append(features)
            
            # get current object position
            curr_obj_pos = np.array([
                time_dict[curr_idx]["obj_state"]["position"]["x"],
                time_dict[curr_idx]["obj_state"]["position"]["y"],
                time_dict[curr_idx]["obj_state"]["position"]["z"]
            ])
            
            # get next object position
            next_obj_pos = np.array([
                time_dict[next_idx]["obj_state"]["position"]["x"],
                time_dict[next_idx]["obj_state"]["position"]["y"],
                time_dict[next_idx]["obj_state"]["position"]["z"]
            ])
            
            # get end effector position
            curr_ee_pos = np.array([
                time_dict[curr_idx]["ee_state"]["x"],
                time_dict[curr_idx]["ee_state"]["y"]
            ])
            
            next_ee_pos = np.array([
                time_dict[next_idx]["ee_state"]["x"],
                time_dict[next_idx]["ee_state"]["y"]
            ])
            
            # convert to relative coordinates - end effector relative to current object position
            rel_ee_x = curr_ee_pos - curr_obj_pos[:2]  
            rel_ee_nx = next_ee_pos - next_obj_pos[:2]  # only x,y coordinates
            
            is_safe = time_dict[curr_idx]["is_safe"]
            if is_safe:
                safe_local_xs.append(local_x)
                safe_local_nxs.append(local_nx)
                safe_ee_xs.append(rel_ee_x)
                safe_ee_nxs.append(rel_ee_nx)
            else:
                unsafe_local_xs.append(local_x)
                unsafe_ee_xs.append(rel_ee_x)
    
    
    return {
        "safe_local_xs": np.array(safe_local_xs, dtype=object),
        "safe_local_nxs": np.array(safe_local_nxs, dtype=object),
        "safe_ee_xs": np.array(safe_ee_xs),
        "safe_ee_nxs": np.array(safe_ee_nxs),
        "unsafe_local_xs": np.array(unsafe_local_xs, dtype=object),
        "unsafe_ee_xs": np.array(unsafe_ee_xs)
    }

def count_nonzero_safety_cost(data):
    """
    count how many safe samples have nonzero safety cost
    
    Args:
        data: dictionary containing reformatted data
        
    Returns:
        total_safe: total number of safe samples
        nonzero_cost: number of safe samples with nonzero safety cost
    """
    safe_local_xs = data["safe_local_xs"]
    total_safe = len(safe_local_xs)
    nonzero_cost = 0
    
    for sample in safe_local_xs:
        # for each sample, check if any history point's tilt_cost is nonzero
        for history_point in sample:
            # the first element of history_point is tilt_cost
            if history_point[0] != 0:
                nonzero_cost += 1
                break
    
    return total_safe, nonzero_cost

def main():
    args = parse_args()
    # convert angle to radians
    max_tilt_rad = math.radians(args.max_tilt_angle)
    
    json_data = load_json_data(args.input)
    organized, large_tilt_count, total_samples = organize_by_object(json_data, max_tilt_rad)
    data = build_sequences(organized, args.horizon, args.step, args.min_seq_len)

    total_safe, nonzero_cost = count_nonzero_safety_cost(data)
    
    # print statistics
    print(f"total samples: {total_samples}")
    print(f"samples with tilt angle greater than {args.max_tilt_angle} degrees: {large_tilt_count}")
    print(f"percentage of samples with tilt angle greater than {args.max_tilt_angle} degrees: {large_tilt_count / total_samples * 100:.2f}%")
    print(f"number of samples retained: {total_samples - large_tilt_count}")
    print(f"\ntotal safe samples: {total_safe}")
    print(f"number of safe samples with nonzero safety cost: {nonzero_cost}")
    print(f"percentage of safe samples with nonzero safety cost: {nonzero_cost / total_safe * 100:.2f}%")
    
    output_path = args.output or args.input.replace(".json", f"_reformat_relative_max_tilt_{args.max_tilt_angle}_six.npz")
    np.savez_compressed(output_path, **data)
    print(f"\nsaved to {output_path}")
    for k, v in data.items():
        print(f"{k}: {v.shape}")

if __name__ == "__main__":
    main()