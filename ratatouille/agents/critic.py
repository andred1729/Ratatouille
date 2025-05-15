import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from ratatouille.networks import MLP

class DoubleQCritic(nn.Module):
    def __init__ (self, observation_dim, action_dim, hidden_dims):
        super(DoubleQCritic, self).__init__()
        self.Q1 = MLP(observation_dim + action_dim, 1, hidden_dims)
        self.Q2 = MLP(observation_dim + action_dim, 1, hidden_dims)
        
    def forward(self, observation, action):
        """Batch processing

        Args:
            observation (torch tensor): observation batch
            action (torch tensor): action batch
        """
        # make sure batch of same size
        assert observation.size(0) == action.size(0)
        observation_action = torch.cat([observation, action], dim=-1)
        q1 = self.Q1(observation_action)
        q2 = self.Q2(observation_action)
        
        return q1, q2