import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ratatouille.utils import to_np
import math


class SACAgent():
    def __init__(self, observation_dim, action_dim, action_range, device, critic_cls, actor_cls, discount, init_temperature, actor_update_frequency, target_critic_update_frequency, alpha_lr, alpha_betas, actor_lr, actor_betas, critic_lr, critic_betas, critic_tau, batch_size, critic_kwargs, actor_kwargs):
        self.action_range = action_range
        self.discount = discount
        self.device = device
        self.critic_tau = critic_tau
        self.batch_size = batch_size
        
        self.actor_update_frequency = actor_update_frequency
        self.target_critic_update_frequency = target_critic_update_frequency

        self.critic = critic_cls(**critic_kwargs).to(self.device)
        self.target_critic = critic_cls(**critic_kwargs).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor = actor_cls(**actor_kwargs).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        self.target_entropy = -action_dim

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

    def update_critic(self, observation, action, reward, next_observation, not_terminal):
        """Update the batch

        Args:
            observation (torch tensor): observation batch
            action (torch tensor): action batch
            reward (torch tensor): reward batch
            next_observation (torch tensor): next observation batch
            not_terminal (torch tensor): not terminal batch
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

        # if not_terminal -> discount with next state
        # if terminal, then done
        target_Q = reward + (not_terminal * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(observation, action)

        # define loss and gradient descent
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, observation):
        # actor
        dist = self.actor(observation)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(observation, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        # want to maximize Q + alpha * entropy = Q - alpha * log$
        # so minimize the negative
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha
        # want entropy -log_prob to be equal to target entropy
        # simple chain rule to make it go up/down depending on overshoot/undershoot
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
    def update(self, observation_batch, action_batch, reward_batch, next_observation_batch, not_terminal_batch, step):
        self.update_critic(observation_batch, action_batch, reward_batch, next_observation_batch, not_terminal_batch)
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(observation_batch)

        if step % self.target_critic_update_frequency == 0:
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.critic_tau * param.data + (1.0 - self.critic_tau) * target_param.data)
        
