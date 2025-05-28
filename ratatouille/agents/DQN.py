import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ratatouille.networks import MLP

class QNetwork(nn.Module):
    """Simple MLP values."""
    def __init__(self, observation_dim, n_actions, hidden_dims):
        super().__init__()
        self.net = MLP(observation_dim, n_actions, hidden_dims)

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, device, buffer, hidden_dims=(256,256,256),
                 gamma=0.99, lr=3e-4, batch_size=256,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=1e5,
                 target_update_freq=1000):
        self.env = env
        self.device = device
        self.replay_buffer = buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = env.n_actions
        self.step_count = 0
        # Networks ---------------------------------------------------------
        self.q_net = QNetwork(env.observation_dim, self.n_actions, hidden_dims).to(device)
        self.target_q_net = QNetwork(env.observation_dim, self.n_actions, hidden_dims).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        # Epsilon schedule --------------------------------------------------
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.target_update_freq = target_update_freq

    # ---------------------------------------------------------------------
    def epsilon(self):
        """Linearly annealed epsilon."""
        progress = min(1.0, self.step_count / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1 - progress)

    @torch.no_grad()
    def act(self, obs, eval_mode=False):
        if (not eval_mode) and (np.random.rand() < self.epsilon()):
            return np.random.randint(self.n_actions)
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        q_values = self.q_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    # ---------------------------------------------------------------------
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}
        obs, actions, rewards, next_obs, discounts = self.replay_buffer.sample(self.batch_size)
        # Q(s,a)
        q_pred = self.q_net(obs).gather(1, actions.unsqueeze(1))
        # target: r + γ * max_a' Q′(s', a')
        with torch.no_grad():
            q_next = self.target_q_net(next_obs).max(1, keepdim=True)[0]
            q_target = rewards + discounts * self.gamma * q_next
        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        # periodic target sync
        if self.step_count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        return {"dqn/loss": loss.item(), "dqn/epsilon": self.epsilon()}