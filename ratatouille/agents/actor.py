import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from ratatouille.networks import MLP
import torch
from torch import distributions as pyd

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims):
        super().__init__()
        # 1 of the action_dims for the mu
        # the other one for log_std
        self.trunk = MLP(
            input_dim=observation_dim,
            output_dim=action_dim * 2,
            hidden_dims=hidden_dims
        )

    def forward(self, observation_batch):
        out = self.trunk(observation_batch)
        assert torch.all(torch.isfinite(out)), "Trunk output contains NaN or Inf"
        mu, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-5, max=2)
        std = log_std.exp()
        
        return SquashedNormal(mu, std)
