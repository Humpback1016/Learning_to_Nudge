import copy
import math
import argparse
import numpy as np
import pickle
from collections import deque
import torch
import os

from config import *
from helpers import *  # this code only uses the normalize_data function
from models import NCBF, Normalizer  

import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--init', action='store_true', help='initialize training model')
args = parser.parse_args()

def load_formatted_data(data_filepath):

    print(f"Loading formatted training data from {data_filepath}...")
    data = np.load(data_filepath, allow_pickle=True)
    
    safe_local_xs = data['safe_local_xs']
    safe_local_nxs = data['safe_local_nxs']
    safe_ee_xs = data['safe_ee_xs']
    safe_ee_nxs = data['safe_ee_nxs']
    unsafe_local_xs = data['unsafe_local_xs']
    unsafe_ee_xs = data['unsafe_ee_xs']
    
    print(f"Loaded training data:")
    print(f"  safe_local_xs shape: {safe_local_xs.shape}")
    print(f"  safe_local_nxs shape: {safe_local_nxs.shape}")
    print(f"  safe_ee_xs shape: {safe_ee_xs.shape}")
    print(f"  safe_ee_nxs shape: {safe_ee_nxs.shape}")
    print(f"  unsafe_local_xs shape: {unsafe_local_xs.shape}")
    print(f"  unsafe_ee_xs shape: {unsafe_ee_xs.shape}")
    
    # 提取特征数据 (只取前4个元素: tilt, rel_x, rel_y, rel_z)
    print("Extracting features for model training...")
    
    # 处理safe_local_xs
    safe_h_obj_features = np.zeros((safe_local_xs.shape[0], safe_local_xs.shape[1], 4), dtype=np.float32)
    for i in range(safe_local_xs.shape[0]):
        for j in range(safe_local_xs.shape[1]):
            safe_h_obj_features[i, j] = safe_local_xs[i, j, :4].astype(np.float32)
    
    # 处理safe_local_nxs
    safe_h_obj_nxs_features = np.zeros((safe_local_nxs.shape[0], safe_local_nxs.shape[1], 4), dtype=np.float32)
    for i in range(safe_local_nxs.shape[0]):
        for j in range(safe_local_nxs.shape[1]):
            safe_h_obj_nxs_features[i, j] = safe_local_nxs[i, j, :4].astype(np.float32)
            
    # 处理unsafe_local_xs
    unsafe_h_obj_features = np.zeros((unsafe_local_xs.shape[0], unsafe_local_xs.shape[1], 4), dtype=np.float32)
    for i in range(unsafe_local_xs.shape[0]):
        for j in range(unsafe_local_xs.shape[1]):
            unsafe_h_obj_features[i, j] = unsafe_local_xs[i, j, :4].astype(np.float32)
    
    # 确保末端执行器数据是浮点类型
    safe_ee_xs = safe_ee_xs.astype(np.float32)
    safe_ee_nxs = safe_ee_nxs.astype(np.float32)
    unsafe_ee_xs = unsafe_ee_xs.astype(np.float32)
    
    print(f"Extracted feature shapes:")
    print(f"  safe_h_obj_features shape: {safe_h_obj_features.shape}")
    print(f"  safe_h_obj_nxs_features shape: {safe_h_obj_nxs_features.shape}")
    print(f"  unsafe_h_obj_features shape: {unsafe_h_obj_features.shape}")
    
    return safe_h_obj_features, safe_h_obj_nxs_features, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_features, unsafe_ee_xs
    
if __name__ == "__main__":
    print("Hello World\nNCBF Arm training starts.")
    print("-------------------------------\n")
    
    # specify formatted data file path
    formatted_data_path = "/workspace/isaaclab/ncbf_trajectories_0108/trajectories_reformat_relative_max_tilt_30.0_one.npz"
    # formatted_data_path = "/workspace/isaaclab/ncbf_trajectories_best/trajectories_reformat_relative.npz"
    
    # load and extract feature data
    safe_h_obj_xs, safe_h_obj_nxs, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_xs, unsafe_ee_xs = load_formatted_data(formatted_data_path)
    
    # normalize the data
    print("\nNormalizing the data...")
    h_obj_inputDim = 4  # fixed to 4 (tilt, rel_x, rel_y, rel_z)
    ee_inputDim = 2  # fixed to 2 (x, y)
    
    h_obj_normalizer = Normalizer(h_obj_inputDim)
    ee_normalizer = Normalizer(ee_inputDim)
    
    # directly use the original normalize_data function
    data = normalize_data(
        safe_h_obj_xs, safe_h_obj_nxs, unsafe_h_obj_xs,
        safe_ee_xs, safe_ee_nxs, unsafe_ee_xs,
        h_obj_normalizer, ee_normalizer, update=True
    )
    
    h_obj_normalizer.save_model(h_obj_normalizer_path)
    ee_normalizer.save_model(ee_normalizer_path)
    print("Normalizing finished.")
    
    # build model - using PyTorch version
    model = NCBF(
        seqInputHorizon=Horizon, 
        seqInputSize=h_obj_inputDim, 
        inputSize=ee_inputDim,
        hiddens=Hiddens, 
        seq_hiddens=SeqHiddens,
        activation="relu",
        regularizer_factor=RegFactor,
        CBF_gamma=Gamma
    )
    
    os.makedirs(os.path.dirname(initial_model_path), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n---------Initial phase starts----------")
    
    # using PyTorch version of fit method, no compile
    model.fit(
        data=data,  # using normalized data
        epoch=Epoch, 
        verbose_num=100,
        lr=LR,
        margin_threshold=MarginThreshold,
        derivative_threshold=DerivativeThreshold,
        save_path=os.path.dirname(initial_model_path),
        save_iters=Epoch//100,
        batch_size=256,   
        device=device     
    )
    
    # save final model
    model.save(initial_model_path)
    print(f"\nfinished. Model saved at {initial_model_path}")
    print("-----------------------------------------\n")
