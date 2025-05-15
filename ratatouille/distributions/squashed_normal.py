import torch
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

class SquashedNormal(TransformedDistribution):
    """
        Squashed Normal distribution using a Tanh transformation.
    """

    def __init__(self, mu, std):
        self.base_dist = Normal(mu, std)
        super().__init__(self.base_dist, TanhTransform())