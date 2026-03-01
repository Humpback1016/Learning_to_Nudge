# Learning to Nudge

This repository contains code for learning Neural Control Barrier Functions (NCBF) for safe robot manipulation: simulation data collection and training (Isaac Sim), refinement from safe/unsafe sets, real-robot evaluation with Franka + Kinect + AprilTag fusion, and CBF visualization.

---

## Directory structure

| Directory | Purpose |
|-----------|--------|
| `evaluation_hardware/` | Real-robot: Franka TF, AprilTag detection/fusion, NCBF control and trajectory recording |
| `evaluation_sim/` | Isaac Sim evaluation and CBF plotting |
| `data_collection/` | Isaac Sim data collection and JSON→NPZ reformatting for training |
| `initial_training/` | NCBF model definition, config, helpers, and initial training script |
| `refinement_training/` | Batch refinement training from unsafe-set trajectories in Isaac Sim |

---

## File index (description + run command)

### evaluation_hardware/

| File | Description | Run command |
|------|-------------|-------------|
| `franka_send_tf.py` | Publishes Franka end-effector pose as TF from `panda_link0` to `panda_EE` at 100 Hz (for hand–eye calibration and NCBF). | `python3 franka_send_tf.py [--host 192.168.1.116]` (with ROS2 and `frankx` env). |
| `dual_kinect_detector.py` | ROS2 node: detects AprilTags in one Kinect RGB stream, publishes per-tag poses and PoseArray; optional video recording. | `python3 dual_kinect_detector.py --ros-args -p camera_name:=kinect2 -p frame_id:=camera2_color_optical_frame` (and similarly for kinect1). |
| `dual_fusion.py` | ROS2 node: fuses AprilTag poses from two Kinects and publishes a single unified pose per tag with quality/timeout logic. | `python3 dual_fusion.py --ros-args -p tag_ids:=[0,2]` |
| `apriltag_detector.py` | Single-camera AprilTag detector: subscribes to one Kinect RGB stream, publishes pose array and TF frames. | Run as ROS2 node (e.g. `python3 apriltag_detector.py`); expects `kinect1/rgb/image_raw` and `kinect1/rgb/camera_info`. |
| `franka_dcbf_ros2_recording_transform.py` | NCBF-guided Franka controller: reads fused AprilTag poses via ROS2, runs NCBF, sends motions via Frankx, records trajectories to JSON. | `python3 franka_dcbf_ros2_recording_transform.py [--host 192.168.1.116] [--max_steps 200] [--velocity 0.01] [--tag_ids 2] [--output_dir real_robot_trajectories]` |

### evaluation_sim/

| File | Description | Run command |
|------|-------------|-------------|
| `reobot_evaluation/main.py` | Isaac Sim launcher for NCBF-guided evaluation: loads model, runs env, records trajectories and optional video. | `./isaaclab.sh -p <path>/main.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 [--num_envs 20] [--record_video] [--model_path ...] [--h_obj_normalizer_path ...] [--ee_normalizer_path ...]` (run from Isaac Lab workspace). |
| `reobot_evaluation/evaluator.py` | Evaluation logic: NCBF-based control, goal pushing, trajectory collection, video recording. | Used by `main.py` (no standalone run). |
| `reobot_evaluation/data_handler.py` | Saves/loads run data and trajectory stats for evaluation. | Used by `main.py` (no standalone run). |
| `reobot_evaluation/stats_analyzer.py` | Computes evaluation statistics (safety violations, tilt, distances, etc.) from run data. | Used by `main.py` (no standalone run). |
| `cbf_test_real.py` | Loads NCBF model and trajectory JSON, generates CBF classification/contour plots (safe vs unsafe regions). | `python3 cbf_test_real.py` (set `trajectory_file`, `model_path`, `h_obj_normalizer_path`, `ee_normalizer_path` in script; run from dir where `models` is importable, e.g. add `initial_training` to PYTHONPATH). |

### data_collection/

| File | Description | Run command |
|------|-------------|-------------|
| `ncbf_manip_dis_random_vel.py` | Isaac Sim auto-exploration: random-velocity pushing with safety, collects trajectories and saves JSON via `collect_data`. | `./isaaclab.sh -p <path>/ncbf_manip_dis_random_vel.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 [--num_envs 40] [--max_steps ...] [--total_runs ...]` |
| `collect_data.py` | Helpers to record robot/object states, safety costs, joint states into trajectories and save to disk; used by `ncbf_manip_dis_random_vel.py`. | Imported by data-collection script (no standalone run). |
| `data_reformat.py` | Converts trajectory JSON to NCBF training format (sliding-window sequences, tilt filtering) and writes NPZ. | `python3 data_reformat.py [--input ncbf_trajectories/trajectories.json] [--output ...] [--horizon 6] [--max_tilt_angle 30.0]` |

### initial_training/

| File | Description | Run command |
|------|-------------|-------------|
| `training.py` | Loads reformatted NPZ, normalizes data, trains NCBF (PyTorch), saves model and normalizers. | `python3 training.py [--init]` (data path and config in script/config). |
| `config.py` | Paths, model/normalizer names, horizon, training hyperparameters, arm limits. | Imported by `training.py` and others (no standalone run). |
| `models.py` | PyTorch `NCBF` (LSTM + MLP) and `Normalizer` for object/EE state. | Imported by training, refinement, and evaluation (no standalone run). |
| `helpers.py` | `normalize_data` and `normalize_state_sequence` for training. | Imported by `training.py` (no standalone run). |

### refinement_training/

| File | Description | Run command |
|------|-------------|-------------|
| `get_refined_demonstrations_hz_unsafe_batch_modular.py` | Isaac Sim script: loads trajectories from JSON, runs NCBF in loop, batch refinement (CBF margin/derivative), saves best model. | `./isaaclab.sh -p <path>/get_refined_demonstrations_hz_unsafe_batch_modular.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 [--num_envs 2000] [--json_file ...] [--model_path ...] [--h_obj_normalizer_path ...] [--ee_normalizer_path ...] [--cbf_min ...] [--cbf_max ...]` (run from Isaac Lab; ensure `models` from initial_training is on PYTHONPATH). |

---

## Real robot workflow

### Aruco / AprilTag fusion (dual Kinect)

1. Start one detector per Kinect (adjust paths to your workspace):

   ```bash
   python3 dual_kinect_detector.py --ros-args -p camera_name:=kinect2 -p frame_id:=camera2_color_optical_frame
   python3 dual_kinect_detector.py --ros-args -p camera_name:=kinect1 -p frame_id:=camera1_color_optical_frame
   ```

2. Start fusion (tag IDs must match your markers):

   ```bash
   python3 dual_fusion.py --ros-args -p tag_ids:=[0,2]
   ```

### Libfranka realtime kernel

If you see “running kernel does not have realtime capabilities”:

```bash
sudo kernelstub -k /boot/vmlinuz-5.15.0-1032-realtime -i /boot/initrd.img-5.15.0-1032-realtime
```

### Hand–eye calibration

1. Start Franka (frankx), Kinect (kinect_ws), and aruco_detector (kinect_ws).
2. Check `/tf` and: `ros2 run tf2_ros tf2_echo panda_link0 panda_EE`.
3. Run calibration (outside conda): `ros2 launch aruco_pkg franka_kinect_calibration.launch.py`.

### Synchronization check (ROS2)

Before experiments, test communication:

- Right terminal: `ros2 topic pub /my_string_topic std_msgs/msg/String 'data:"Hello!"'`
- Left terminal: `ros2 topic list` / `ros2 topic echo /my_string_topic`  
If this fails, check network/ROS_DOMAIN_ID and wiring.

### Activate Franka

1. Check control button and emergency stop; confirm correct button.
2. Connect to robot (e.g. 192.168.1.116/desk), unlock joints, press control button, set pose. Yellow = connecting, blue = ready, white = control button pressed.
3. Example (adjust IP and paths):

   ```bash
   conda activate frankx
   cd /workspaces/frankx/example
   # Ensure IP in code is correct (e.g. 192.168.1.116)
   python linear.py
   source /opt/ros/humble/setup.bash
   ```

### Activate Kinect

```bash
export ROS_DOMAIN_ID=50
conda activate kinect_ws
cd workspaces/ros2_kinect/src/Azure_Kinect_ROS_Driver/
source ../../install/setup.bash
ros2 launch azure_kinect_ros_driver driver.launch.py
```

Then run AprilTag detector (e.g. `detector_node_multi.py` or `apriltag_detector.py` from this repo as needed).

### Useful test commands

```bash
ros2 run rqt_image_view rqt_image_view
rviz2
ros2 topic list
ros2 topic echo /camera/color/camera_info
ros2 topic echo /camera/color/image_raw
# With serial number for a specific device:
ros2 launch azure_kinect_ros_driver driver.launch.py cam1_sensor_sn:=000152922712
```

---

## Isaac Sim / Isaac Lab workflow

- **Merge data:** `python3 scripts/environments/teleoperation/merge_json.py` (if that script exists in your Isaac Lab tree).
- **Collect data:**  
  `./isaaclab.sh -p scripts/environments/teleoperation/ncbf_manip_dis_random_vel.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 40`  
  Tune `--max_steps`, `--total_runs`, `--num_envs` as needed.
- **Reformat for training:**  
  `./isaaclab.sh -p scripts/environments/teleoperation/data_reformat.py`  
  Adjust input trajectory path and options (e.g. `--horizon`, `--max_tilt_angle`) in the script or CLI.
- **Refinement from unsafe set:**  
  `./isaaclab.sh -p scripts/environments/teleoperation/get_refined_demonstrations_hz_unsafe.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0`  
  Tune `--num_envs`, `--json_file`, `--model_path`, normalizer paths, `--cbf_min`/`--cbf_max`, horizon, and `num_move_steps` in `process_batch` if needed.
- **Refinement from safe set:**  
  `./isaaclab.sh -p scripts/environments/teleoperation/get_refined_demonstrations_hz.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0`  
  Same parameter notes; set safety boundary using CBF value plots.
- **Evaluation (NCBF):**  
  `./isaaclab.sh -p scripts/environments/teleoperation/inference_multi.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 [--num_envs 20]`  
  For video: add `--record_video --enable_cameras` and `--num_envs 1`. Set `--h_obj_normalizer_path`, `--ee_normalizer_path`, `--model_path`, `--max_steps` as needed.
- **Baselines:**  
  - APF: `./isaaclab.sh -p scripts/environments/teleoperation/baseline_apf.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 20`  
  - Naive step backward: `./isaaclab.sh -p scripts/environments/teleoperation/baseline_naive_backward.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 20`  
  - “Do nothing”: use `baseline_naive_backward.py` and uncomment the block from the `if safe_mode:` line to `exploration_data['ncbf_preventions'] += 1` (as in your notes).

For all evaluation scripts, set `base_dir` in `save_run_stats` and adjust `--max_steps`, `--num_envs`, `--total_runs` as needed.

---

## References

- [Azure Kinect multi-camera sync](https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/multi-camera-sync)
- [Hand–eye calibration](https://github.com/marcoesposito1988/easy_handeye2)
