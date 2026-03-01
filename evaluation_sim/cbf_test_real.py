import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json
from models import NCBF, Normalizer
from collections import deque, defaultdict
from tqdm import tqdm
from matplotlib.patches import Circle

initial_model_path = "boundary_check_results/model_batch_11.pt"
# initial_model_path = "results/model_iter_40.pt" 
h_obj_normalizer_path = "results_0202_1/local_normalizer"  
ee_normalizer_path = "results_0202_1/arm_normalizer"      
output_dir = "cbf_classification_plots"
os.makedirs(output_dir, exist_ok=True)

def load_model_and_normalizers(model_path, h_obj_normalizer_path, ee_normalizer_path):

    model = NCBF.load(
        path=model_path,
        hiddens=[64, 64],  # hidden layer sizes
        seq_hiddens=[64]   # sequence hidden layer sizes
    )
    
    # load object state normalizer (4 features: tilt, rel_x, rel_y, rel_z)
    h_obj_normalizer = Normalizer(input_size=4)
    h_obj_normalizer.load_model(h_obj_normalizer_path)
    
    # load end-effector normalizer (2 features: ee_x, ee_y)
    ee_normalizer = Normalizer(input_size=2)
    ee_normalizer.load_model(ee_normalizer_path)
    
    return model, h_obj_normalizer, ee_normalizer

def load_trajectory_data(json_file_path):

    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    
    # organize data by object
    organized = defaultdict(lambda: defaultdict(list))
    for t, entry in enumerate(json_data):
        objects = entry["objects_state"]
        is_safe = entry.get("is_safe", [[True] for _ in objects])
        ee_state = entry["ee_state"]
        for obj_idx, (obj, safe_flag) in enumerate(zip(objects, is_safe)):
            # only process the first boolean value
            safe = safe_flag[0] if isinstance(safe_flag, list) else safe_flag
            
            # get the object name, if not exist, create a default name
            obj_name = obj.get("name", f"object_{obj_idx}")
            
            data_point = {
                "t": t,
                "ee_state": ee_state,
                "obj_state": obj,
                "is_safe": safe,
                "obj_name": obj_name
            }
            organized[obj_idx][t] = data_point
    
    # convert to trajectory format
    trajectories = []
    for obj_idx, time_dict in organized.items():
        # sort by time
        times = sorted(time_dict.keys())
        obj_trajectory = []
        ee_trajectory = []
        
        for t in times:
            obj_tilt = time_dict[t]["obj_state"]["tilt_cost"]
            obj_pos = np.array([
                time_dict[t]["obj_state"]["position"]["x"],
                time_dict[t]["obj_state"]["position"]["y"],
                time_dict[t]["obj_state"]["position"]["z"]
            ])
            
            ee_pos = np.array([
                time_dict[t]["ee_state"]["x"],
                time_dict[t]["ee_state"]["y"]
            ])
            
            obj_trajectory.append([obj_tilt, obj_pos[0], obj_pos[1], obj_pos[2]])
            ee_trajectory.append(ee_pos)
        
        trajectories.append({
            "obj_trajectory": np.array(obj_trajectory),
            "ee_trajectory": np.array(ee_trajectory),
            "obj_name": time_dict[times[0]]["obj_name"]
        })
    
    return trajectories

def generate_multi_bottle_cbf_map(
    model,
    h_obj_normalizer,
    ee_normalizer,
    trajectories,
    grid_range=0.4,
    resolution=30,
    horizon=3,
    time_step=0
):

    # get all bottle positions at the given time step
    bottle_positions = []
    for traj in trajectories:
        if len(traj["obj_trajectory"]) <= time_step:
            continue  # skip objects with insufficient trajectory length
        bottle_pos = traj["obj_trajectory"][time_step][1:4]  # skip tilt_cost, only take xyz
        bottle_positions.append(bottle_pos)
    
    # determine the boundary of the overall map
    all_x = [pos[0] for pos in bottle_positions]
    all_y = [pos[1] for pos in bottle_positions]
    min_x, max_x = min(all_x) - grid_range, max(all_x) + grid_range
    min_y, max_y = min(all_y) - grid_range, max(all_y) + grid_range
    
    # create the grid of the overall map
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # collect the CBF values of each bottle
    all_cbf_arrays = []
    
    # device settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # ensure the model is in evaluation mode
    
    # generate the CBF map for each bottle
    for traj_idx, traj in enumerate(trajectories):
        obj_trajectory = traj["obj_trajectory"]
        
        # check the trajectory length
        if len(obj_trajectory) <= time_step or len(obj_trajectory) < horizon:
            print(f"skip trajectory {traj_idx}: length insufficient (need at least {max(time_step+1, horizon)} frames, actual {len(obj_trajectory)} frames)")
            continue
        
        # adjust the time step to ensure there is enough history
        adjusted_time_step = max(horizon-1, time_step)
        
        # get the history window (the last horizon frames)
        history_window = obj_trajectory[adjusted_time_step-horizon+1:adjusted_time_step+1]
        
        # check the history window length
        if len(history_window) != horizon:
            print(f"warning: history window length ({len(history_window)}) does not match the expected time span ({horizon})")
            # if the history window length is insufficient, copy the last frame to fill
            while len(history_window) < horizon:
                history_window = np.vstack([history_window[0:1], history_window])
        
        # calculate the history relative to the base point
        base_pos = history_window[0][1:4]  # the first history point as the base point
        
        # create the relative history window
        rel_history = []
        for point in history_window:
            obj_tilt = point[0]  # tilt_cost
            obj_pos = point[1:4]  # xyz
            rel_pos = obj_pos - base_pos
            rel_history.append([obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]])
        
        rel_history = np.array(rel_history)
        
        # print the history window shape for debugging
        print(f"bottle {traj_idx} history window shape: {rel_history.shape}")
        
        # initialize the CBF value matrix for the current bottle
        bottle_cbf_values = np.zeros((resolution, resolution))
        
        # calculate the CBF value for each grid point
        for i in tqdm(range(resolution), desc=f"processing bottle {traj_idx+1}/{len(trajectories)}"):
            for j in range(resolution):
                # the current grid point in the global coordinate system
                grid_x = xx[i, j]
                grid_y = yy[i, j]
                
                # calculate the position of the robot relative to the base point
                rel_robot_pos = np.array([grid_x, grid_y]) - base_pos[:2]
                
                # ensure the input dimension is correct
                if rel_history.shape[1] != 4:
                    print(f"error: history window feature dimension ({rel_history.shape[1]}) does not match the expected 4")
                    continue
                
                # apply normalization - this is a new part
                normalized_rel_history = h_obj_normalizer(rel_history, update=False)
                normalized_rel_robot_pos = ee_normalizer(np.array([rel_robot_pos]), update=False)
                
                # convert to PyTorch tensor
                rel_history_tensor = torch.tensor(normalized_rel_history, dtype=torch.float32, device=device).unsqueeze(0)
                rel_robot_pos_tensor = torch.tensor(normalized_rel_robot_pos, dtype=torch.float32, device=device)
                
                # check the tensor shape
                if rel_history_tensor.dim() != 3 or rel_history_tensor.size(2) != 4:
                    print(f"error: history window tensor shape is incorrect: {rel_history_tensor.shape}, expected (1, {horizon}, 4)")
                    continue
                
                # predict the CBF value - use torch.no_grad() to avoid calculating gradients
                with torch.no_grad():
                    cbf_value = model([rel_history_tensor, rel_robot_pos_tensor]).cpu().detach().numpy()[0]
                    bottle_cbf_values[i, j] = cbf_value
        
        all_cbf_arrays.append(bottle_cbf_values)
    
    # use the maximum operation to combine the CBF values of all bottles
    final_cbf_values = np.max(all_cbf_arrays, axis=0)
    
    return xx, yy, final_cbf_values

def plot_multi_bottle_cbf_map(xx, yy, cbf_values, trajectories, time_step, save_path=None):
    """plot the CBF value distribution of multiple bottles"""
    plt.figure(figsize=(12, 10))
    
    # plot the filled contour of the CBF values
    contour = plt.contourf(xx, yy, cbf_values, 20, cmap='RdBu_r', alpha=0.8)
    plt.colorbar(contour, label='CBF values')
    
    # plot the zero contour (CBF=0 boundary)
    zero_contour = plt.contour(
        xx, yy, cbf_values, levels=[0],
        colors='black', linewidths=2, linestyles='dashed'
    )
    # plt.clabel(zero_contour, inline=True, fontsize=10, fmt='%1.1f')
    
    # mark the bottle positions - use semi-transparent circles, radius 5 cm
    for i, traj in enumerate(trajectories):
        if len(traj["obj_trajectory"]) <= time_step:
            continue  # skip objects with insufficient trajectory length
        
        bottle_pos = traj["obj_trajectory"][time_step][1:4]  # skip tilt_cost, only take xyz
        
        # create a semi-transparent circle, radius 5 cm
        circle = Circle(
            (bottle_pos[0], bottle_pos[1]),
            radius=0.07,  # 5 cm = 0.05 m
            color='brown',
            alpha=0.5,  # set the transparency
            label=f'bottle {i+1}' if i == 0 else ""
        )
        plt.gca().add_patch(circle)
        
        # add the center mark
        plt.plot(bottle_pos[0], bottle_pos[1], 'k+', markersize=5)
        
        # plot the end-effector position
        if len(traj["ee_trajectory"]) > time_step:
            ee_pos = traj["ee_trajectory"][time_step]
            # use a red star to mark the end-effector position
            plt.plot(ee_pos[0], ee_pos[1], 'r*', markersize=10, 
                    label='end-effector' if i == 0 else "")
    
    # chart properties
    if trajectories and len(trajectories[0]["obj_trajectory"]) > time_step:
        tilt_degrees = np.degrees(trajectories[0]["obj_trajectory"][time_step][0])
        tilt_info = f'tilt angle: {tilt_degrees:.1f}°'
    else:
        tilt_info = ""
    
    plt.title(
        f'CBF value distribution - multiple bottles\n'
        f'time step: {time_step} {tilt_info}'
    )
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # axis range
    plt.xlim([xx.min(), xx.max()])
    plt.ylim([yy.min(), yy.max()])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"chart saved to: {save_path}")
    
    # close the chart to avoid memory leak
    plt.close()

def generate_trajectory_cbf_plots(trajectory_file, model_path, h_obj_normalizer_path, ee_normalizer_path):

    model, h_obj_normalizer, ee_normalizer = load_model_and_normalizers(
        model_path, h_obj_normalizer_path, ee_normalizer_path
    )
    
    # load the trajectory data
    trajectories = load_trajectory_data(trajectory_file)
    
    # determine the maximum available time step
    max_available_step = min(len(traj["obj_trajectory"]) for traj in trajectories) - 1
    print(f"maximum available time step: {max_available_step}")
    
    # select one time step every 100 time steps for visualization
    # ensure it does not exceed the maximum available step
    time_steps = list(range(0, max_available_step + 1, 100))
    if not time_steps:  # ensure there is at least one time step
        time_steps = [0]
    if max_available_step not in time_steps:  # add the last time step
        time_steps.append(max_available_step)
    
    print(f"will generate charts for the following time steps: {time_steps}")
    
    # generate charts for each time step
    for time_step in time_steps:
        print(f"generate chart for the time step {time_step} of multiple bottles")
        
        xx, yy, cbf_values = generate_multi_bottle_cbf_map(
            model,
            h_obj_normalizer,
            ee_normalizer,
            trajectories=trajectories,
            grid_range=1.0,  # adjust the grid range to fit multiple bottles
            resolution=50,   # increase the resolution to get a smoother result
            horizon=3,
            time_step=time_step
        )
        
        filename = f"multi_bottle_cbf_timestep_{time_step}.png"
        save_path = os.path.join(output_dir, filename)
        
        plot_multi_bottle_cbf_map(
            xx, yy, cbf_values,
            trajectories, time_step,
            save_path=save_path
        )
        


if __name__ == "__main__":

    trajectory_file = "ncbf_trajectories_test/trajectories.json"  
    model_path = initial_model_path
    
    generate_trajectory_cbf_plots(
        trajectory_file, 
        model_path, 
        h_obj_normalizer_path, 
        ee_normalizer_path
    )