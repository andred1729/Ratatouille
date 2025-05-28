import numpy as np
import torch

class ReplayBufferDQN:
    """Uniform‑sampling replay buffer (no PER)."""
    def __init__(self, observation_dim: int, capacity: int, device):
        self.capacity = capacity
        self.device = device
        self.obs_buf = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_obs_buf = np.empty((capacity, observation_dim), dtype=np.float32)
        self.action_buf = np.empty((capacity,), dtype=np.int64)
        self.reward_buf = np.empty((capacity,), dtype=np.float32)
        self.discount_buf = np.empty((capacity,), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def __len__(self):
        return self.size

    def insert(self, obs, action, reward, next_obs, discount):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.discount_buf[self.ptr] = discount

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idx], device=self.device).float()
        next_obs = torch.as_tensor(self.next_obs_buf[idx], device=self.device).float()
        action = torch.as_tensor(self.action_buf[idx], device=self.device).long()
        reward = torch.as_tensor(self.reward_buf[idx], device=self.device).unsqueeze(1)
        discount = torch.as_tensor(self.discount_buf[idx], device=self.device).unsqueeze(1)
        return obs, action, reward, next_obs, discount