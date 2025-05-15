import numpy as np

import torch
import math
from torch import nn
import torch.nn.functional as F
from ratatouille.distributions import SquashedNormal
from ratatouille.networks import MLP


class DiagGaussianActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims):
        super().__init__()
        self.trunk = MLP(
            input_dim=observation_dim,
            output_dim=action_dim * 2,
            hidden_dims=hidden_dims
        )

    def forward(self, observation_batch):
        mu, log_std = torch.chunk(self.trunk(observation_batch), 2, dim=-1)
        log_std = torch.clamp(log_std, min=-5, max=2)
        std = log_std.exp()
        return SquashedNormal(mu, std)
