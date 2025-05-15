import numpy as np
import torch
from ratatouille.config import rng
class ReplayBuffer():
    """
    Buffer to store environment transitions
    
    Actually don't care about truncated
    
    Only care about terminal for discounting
    
    Use observation and state interchangeably
    
    Store (observation, next_observations, action, reward, terminal, not_terminal)
    
    Only returns (observation, next_observations, action, reward, not_terminal) because discount has same information as terminal
    """
    def __init__(self, observation_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.terminals = np.empty((capacity, 1), dtype=bool)
        self.not_terminals = np.empty((capacity, 1), dtype=np.float32)
        self.insert_index = 0
        self.size = 0
        
    def __len__(self):
        return self.size
    
    def insert(self, observation, action, reward, next_observation, terminal, truncated):
        np.copyto(self.observations[self.insert_index], observation)
        np.copyto(self.actions[self.insert_index], action)
        np.copyto(self.rewards[self.insert_index], reward)
        np.copyto(self.next_observations[self.insert_index], next_observation)
        np.copyto(self.terminals[self.insert_index], terminal)
        np.copyto(self.not_terminals[self.insert_index], 1.0 - float(terminal))
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        idxs = rng.randint(0, self.size, size=batch_size)
        observations = torch.as_tensor(self.observations[idxs], device = self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device = self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        next_observations = torch.as_tensor(self.next_observations[idxs], device=self.device).float()
        not_terminals = torch.as_tensor(self.not_terminals[idxs], device = self.device).float()
        
        return observations, actions, rewards, next_observations, not_terminals