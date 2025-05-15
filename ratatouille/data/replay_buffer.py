import numpy as np
import torch

class ReplayBuffer():
    """
    Buffer to store environment transitions
    """
    def __init__(self, observation_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        
        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        
        self.insert_index = 0
        self.size = 0
        
    def __len__(self):
        return self.size