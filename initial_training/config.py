import math 

# file path
# arm_trajectories_filepath = "ee_demo/ee_states.npy"  # N, T, x, y, vx, vy
# object_states_filepath = "ee_demo/object_states.npy"        # N, T, tilt angle, rel_x, rel_y, rel_z
# arm_actions_filepath = "ee_demo/ee_actions.npy"            # N, T, velocity

# model path
# dynamics_model_path = "results/arm_dynamics_model"
initial_model_path = "results/trained_NCBF_arm_initial"
# refined_model_path = "results/trained_NCBF_arm_refined"
h_obj_normalizer_path = "results/local_normalizer"
ee_normalizer_path = "results/arm_normalizer"  

# pretrained model path
# pretrained_dynamics_model_path = "pretrained_models/arm_dynamics"
# pretrained_initial_model_path = "pretrained_models/arm_cbf_initial"
# pretrained_refined_model_path = "pretrained_models/arm_cbf" 
# pretrained_local_normalizer_path = "pretrained_models/local_normalizer"
# pretrained_vehicle_normalizer_path = "pretrained_models/arm_normalizer"

# dim config
NumObjects = 4  # num of objects   uslesss
NumObjectsTrain = 1  # num of objects while training
Horizon = 0  # history horizon
ConsiderDistance = 1.0  # consider range(meter)   uslesss
Radius = 0.01  # ee radiius
UnsafeAngle = 15.0  # tilt angle

# training params
LR = 1e-4 
Epoch = 2000
SeqHiddens = [64]  # LSTM hidden size
Hiddens = [64, 64]  # MLP hidden sizes
RegFactor = 0.01 
Gamma = 0.1
MarginThreshold = 0.02
DerivativeThreshold = 0.025

# dynamic model params
# DynamicsHiddens = [64, 64]
# DynamicsEpoch = 200

# other params      useless at initial training
ObjectGoalSampleRange = [0.5, 2.0]  # object goal position sampling range
RefinementBufferSize = 10000
RefinementEpochs = 100
RefinementUpdateEpochs = 1000
RefinementTolerance = 0.05
RefinementNumSamples = 50
RefinementSampleSize = 500

# ee params
ARM_MaxVel = 0.5  # max ee vel
ARM_MaxAcc = 0.3  # max ee acc

DemoSize = int(5e4)
dynamics_model_path_arm = "trained_models/arm_dynamics_model"