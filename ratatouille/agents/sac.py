import numpy as np
import torch
import torch.nn as nn
from absl import logging
import torch.nn.functional as F
from ratatouille.utils import to_np
from ratatouille.agents.critic import DoubleQCritic
from ratatouille.agents.actor import DiagGaussianActor


class SACAgent():
    def __init__(self, env, device, init_temperature=1.0, critic_kwargs={"hidden_dims": (256, 256)}, actor_kwargs={"hidden_dims": (256, 256)}, batch_size=32, critic_cls=DoubleQCritic, actor_cls=DiagGaussianActor, actor_update_frequency=2, target_critic_update_frequency=2, alpha_lr=3e-4, alpha_betas=(0.9, 0.999), actor_lr=3e-4, actor_betas=(0.9, 0.999), critic_lr=3e-4, critic_betas=(0.9, 0.999), critic_tau=0.005):
        self.env = env
        self.observation_dim = env.observation_dim
        self.action_dim = env.action_dim
        self.action_range = env.action_range
        self.device = device
        self.discount = env.discount
        self.init_temperature = init_temperature
        self.critic_tau = critic_tau
        self.batch_size = batch_size

        logging.info(f"Initialized SACAgent with observation_dim={self.observation_dim}, "
                     f"action_dim={self.action_dim}, action_range={self.action_range}, "
                     f"device={self.device}, discount={self.discount}, "
                     f"init_temperature={self.init_temperature}, critic_tau={self.critic_tau}, "
                     f"batch_size={self.batch_size}")
        self.actor_update_frequency = actor_update_frequency
        self.target_critic_update_frequency = target_critic_update_frequency

        self.critic = critic_cls(
            env.observation_dim, env.action_dim, **critic_kwargs).to(self.device)
        self.target_critic = critic_cls(
            env.observation_dim, env.action_dim, **critic_kwargs).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor = actor_cls(env.observation_dim,
                               env.action_dim, **actor_kwargs).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        self.target_entropy = -self.action_dim

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)
        self.train()
        self.target_critic.train()

        logging.info(f"Actor structure: {self.actor}")
        logging.info(f"Critic structure: {self.critic}")
        logging.info(f"Target Critic structure: {self.target_critic}")
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, observation):
        """Act on a single point of observation

        Args:
            observation (1d torch tensor): 1 point of observation
        """
        observation = torch.FloatTensor(observation).to(self.device)
        observation_1_batch = observation.unsqueeze(0)
        dist = self.actor(observation_1_batch)
        action = dist.sample()
        action = action.clamp(*self.action_range)
        return to_np(action[0])

    def update_critic(self, observation, action, reward, next_observation, discount_mask, is_weights):
        """Update the batch

        Args:
            observation (torch tensor): observation batch
            action (torch tensor): action batch
            reward (torch tensor): reward batch
            next_observation (torch tensor): next observation batch
            discount_mask (torch tensor): discount mask batch
        """
        dist = self.actor(next_observation)
        next_action = dist.rsample()

        # calculating entropy
        # sum -1 because multiple action dimension, so total entropy is
        # sum of log on each dimension
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        

        # pessimistic Q update
        target_Q1, target_Q2 = self.target_critic(
            next_observation, next_action)

        # entropy regularization
        target_V = torch.min(target_Q1, target_Q2) - \
            self.alpha.detach() * log_prob

        # if discount_mask -> discount with next state
        # if terminal, then done
        target_Q = reward + (discount_mask * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(observation, action)

        # define loss and gradient descent
        
        td_error_1 = (current_Q1 - target_Q)
        td_error_2 = (current_Q2 - target_Q)
        
        if is_weights is not None:
            critic_loss = (is_weights * td_error_1.pow(2)).mean() + \
                (is_weights * td_error_2.pow(2)).mean()
        else:
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # output td error for PER
        with torch.no_grad():
            td_error = (0.5 * td_error_1.abs() +
                        0.5 * td_error_2.abs()).squeeze()
            td_error_np = to_np(td_error)

        self.critic_optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        critic_loss.backward()

        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss,
        }, td_error_np

    def update_actor_and_alpha(self, observation):
        # actor
        dist = self.actor(observation)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        assert torch.isfinite(log_prob).all(), "log_prob has NaN"

        actor_Q1, actor_Q2 = self.critic(observation, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        assert torch.isfinite(actor_Q1).all(), "Q1 has NaN"
        assert torch.isfinite(actor_Q2).all(), "Q2 has NaN"

        # want to maximize Q + alpha * entropy = Q - alpha * log$
        # so minimize the negative
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        assert torch.isfinite(actor_loss), "actor_loss has NaN"

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # with torch.autograd.detect_anomaly():
        self.actor_optimizer.step()

        # alpha
        # want entropy -log_prob to be equal to target entropy
        # simple chain rule to make it go up/down depending on overshoot/undershoot
        alpha_loss = (self.alpha * (-log_prob -
                      self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return {
            "actor_loss": actor_loss,
            "target_entropy": self.target_entropy,
            "actor_entropy": -log_prob.mean(),
            "alpha_loss": alpha_loss,
            "alpha_value": self.alpha
        }

    def update(self, observation_batch, action_batch, reward_batch, next_observation_batch, discount_mask_batch, step, is_weights=None):
        update_info, td_error_np = self.update_critic(
            observation_batch, action_batch, reward_batch, next_observation_batch, discount_mask_batch, is_weights)
        update_info.update({
            "batch_reward": reward_batch.mean()
        })
        if step % self.actor_update_frequency == 0:
            actor_info = self.update_actor_and_alpha(observation_batch)
            update_info.update(actor_info)

        if step % self.target_critic_update_frequency == 0:
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    self.critic_tau * param.data + (1.0 - self.critic_tau) * target_param.data)

        return update_info, td_error_np
