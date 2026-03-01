from config import *
import math
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def normalize_state_sequence(local_xs, normalizer, update = True):
    flattened_local_xs = copy.deepcopy(local_xs.reshape([-1, local_xs.shape[-1]]))
    if update:
        normalizer.experience(flattened_local_xs)
    normalized = normalizer(flattened_local_xs)
    return flattened_local_xs.reshape(local_xs.shape)

def normalize_data(safe_local_xs, safe_local_nxs, unsafe_local_xs,
                  safe_vehicle_xs, safe_vehicle_nxs, unsafe_vehicle_xs,
                  local_normalizer, vehicle_normalizer, update = False):
    normalized_safe_local_xs = normalize_state_sequence(safe_local_xs, local_normalizer,
                                              update = update).astype(np.float32)
    normalized_safe_local_nxs = normalize_state_sequence(safe_local_nxs, local_normalizer,
                                                  update = False).astype(np.float32)
    normalized_unsafe_local_xs = normalize_state_sequence(unsafe_local_xs, local_normalizer,
                                                    update = False).astype(np.float32)
    
    if update:
        vehicle_normalizer.experience(safe_vehicle_xs)
        vehicle_normalizer.experience(unsafe_vehicle_xs)
    normalized_safe_vehicle_xs = vehicle_normalizer(safe_vehicle_xs,
                                                update = False).astype(np.float32)
    normalized_safe_vehicle_nxs = vehicle_normalizer(safe_vehicle_nxs,
                                                     update = False).astype(np.float32)
    normalized_unsafe_vehicle_xs = vehicle_normalizer(unsafe_vehicle_xs,
                                                      update = False).astype(np.float32)
    
    data = [normalized_safe_local_xs, normalized_safe_local_nxs, normalized_safe_vehicle_xs, normalized_safe_vehicle_nxs, normalized_unsafe_local_xs, normalized_unsafe_vehicle_xs]
    return data

