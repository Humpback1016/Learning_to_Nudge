# Learning to Nudge

Code for learning Neural Control Barrier Functions (NCBF) for safe robot manipulation: simulation data collection and training (Isaac Sim), refinement from safe/unsafe sets, real-robot evaluation with Franka + Kinect + AprilTag fusion, and CBF visualization.

---

## Directory structure

| Directory | Purpose |
|-----------|--------|
| `data_collection/` | Isaac Sim data collection and JSON→NPZ reformatting for training |
| `initial_training/` | NCBF model definition, config, helpers, and initial training |
| `refinement_training/` | Batch refinement training from unsafe-set trajectories in Isaac Sim |
| `evaluation_sim/` | Isaac Sim evaluation and CBF plotting |
| `evaluation_hardware/` | Real-robot: Franka TF, AprilTag detection/fusion, NCBF control and trajectory recording |

---

## File index (brief description)

### data_collection/

| File | Description |
|------|-------------|
| `ncbf_manip_dis_random_vel.py` | Isaac Sim auto-exploration script: random-velocity pushing with safety; collects trajectories and saves JSON via `collect_data`. |
| `collect_data.py` | Helpers to record robot/object states, safety costs, and joint states into trajectories and save to disk; used by the data-collection script. |
| `data_reformat.py` | Converts trajectory JSON to NCBF training format (sliding-window sequences, tilt filtering) and writes NPZ. |

### initial_training/

| File | Description |
|------|-------------|
| `training.py` | Loads reformatted NPZ data, normalizes it, trains the NCBF (PyTorch), and saves model and normalizers. |
| `config.py` | Central config: paths, model/normalizer names, horizon, training hyperparameters, and arm limits. |
| `models.py` | Defines PyTorch `NCBF` (LSTM + MLP) and `Normalizer` for object and end-effector state. |
| `helpers.py` | Provides `normalize_data` and `normalize_state_sequence` for training data normalization. |

### refinement_training/

| File | Description |
|------|-------------|
| `get_refined_demonstrations_hz_unsafe_batch_modular.py` | Isaac Sim script: loads trajectories from JSON, runs NCBF in a loop, performs batch refinement (CBF margin/derivative), and saves the best model. |

### evaluation_sim/

| File | Description |
|------|-------------|
| `reobot_evaluation/main.py` | Isaac Sim launcher for NCBF-guided evaluation: loads model, runs env, records trajectories and optional video. |
| `reobot_evaluation/evaluator.py` | Implements evaluation logic: NCBF-based control, goal pushing, trajectory collection, and video recording. |
| `reobot_evaluation/data_handler.py` | Handles saving/loading of run data and trajectory statistics for evaluation. |
| `reobot_evaluation/stats_analyzer.py` | Computes evaluation statistics (safety violations, tilt, distances, etc.) from run data. |
| `cbf_test_real.py` | Loads NCBF model and trajectory JSON and generates CBF classification/contour plots (safe vs unsafe regions). |

### evaluation_hardware/

| File | Description |
|------|-------------|
| `franka_send_tf.py` | Publishes Franka end-effector pose as TF from `panda_link0` to `panda_EE` at 100 Hz for hand–eye calibration and NCBF. |
| `dual_kinect_detector.py` | ROS2 node that detects AprilTags in one Kinect RGB stream and publishes per-tag poses and PoseArray; supports optional video recording. |
| `dual_fusion.py` | ROS2 node that fuses AprilTag poses from two Kinects and publishes a single unified pose per tag with quality and timeout logic. |
| `apriltag_detector.py` | Single-camera AprilTag detector: subscribes to one Kinect RGB stream and publishes pose array and TF frames. |
| `franka_dcbf_ros2_recording_transform.py` | NCBF-guided Franka controller: reads fused AprilTag poses via ROS2, runs NCBF, sends motions via Frankx, and records trajectories to JSON. |
