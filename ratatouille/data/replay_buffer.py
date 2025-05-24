import numpy as np
import torch
class ReplayBuffer():
    """
    Buffer to store environment transitions
    
    Actually don't care about truncated
    
    Only care about terminal for discounting
    
    Use observation and state interchangeably
    
    Store (observation, next_observations, action, reward, discount_mask)
    
    Only returns (observation, next_observations, action, discount_mask) 
    """
    def __init__(self, observation_dim, action_dim, capacity, device, use_PER = False, alpha=0.3, beta=0.4, beta_annealing=1e-7, eps=1e-6):
        self.capacity = capacity
        self.device = device
        
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.discount_masks = np.empty((capacity, 1), dtype=np.float32)
        self.insert_index = 0
        self.size = 0
        
        # PER settings
        self.use_PER = use_PER
        self.alpha = alpha  # controls degree of prioritization
        self.beta = beta    # controls how much IS correction is used
        self.beta_annealing = beta_annealing
        self.eps = eps      # small constant to avoid 0 priority
        if self.use_PER:
            self.priorities = np.zeros((capacity,), dtype=np.float32)

        
    def __len__(self):
        return self.size
    
    def insert(self, observation, action, reward, next_observation, terminal):
        np.copyto(self.observations[self.insert_index], observation)
        np.copyto(self.actions[self.insert_index], action)
        np.copyto(self.rewards[self.insert_index], reward)
        np.copyto(self.next_observations[self.insert_index], next_observation)
        np.copyto(self.discount_masks[self.insert_index], 1.0 - float(terminal))
        
        if self.use_PER:
            max_priority = self.priorities.max() if self.size > 0 else 1.0
            # new transitions are initialized in the replay buffer with maximum priority
            self.priorities[self.insert_index] = max_priority

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """Returns a batch from replay buffer

        Args:
            batch_size (int): Batch size

        Returns:
            (observations_batch, actions_batch, rewards_batch, next_observations_batch, discount_masks_batch): The sampled batch
        """
        if self.use_PER:
            probs = self.priorities[:self.size] ** self.alpha
            probs /= probs.sum()
            
            idxs = np.random.choice(self.size, batch_size, p=probs)
            weights = (self.size * probs[idxs]) ** (-self.beta)
            weights /= weights.max()
            
            self.beta = min(1.0, self.beta + self.beta_annealing)
            weights = torch.as_tensor(weights, device=self.device).unsqueeze(1).float()
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
            weights = torch.ones((batch_size, 1), device=self.device)
            probs = np.ones((self.size,))/self.size


        observations = torch.as_tensor(self.observations[idxs], device = self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device = self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        next_observations = torch.as_tensor(self.next_observations[idxs], device=self.device).float()
        discount_masks = torch.as_tensor(self.discount_masks[idxs], device = self.device).float()
    
        return observations, actions, rewards, next_observations, discount_masks, weights, idxs, probs
    
    def update_priorities(self, idxs, td_error_np):
        if not self.use_PER:
            return
        
        priorities = (td_error_np + self.eps).squeeze()
        self.priorities[idxs] = priorities