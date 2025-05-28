import numpy as np
from gymnasium import Env, spaces

class DiscreteActionWrapper(Env):
    """Turns RatEnv's 2‑D continuous wheel speeds into a 441‑way discrete space."""
    def __init__(self, rat_env):
        super().__init__()
        self.env = rat_env

        # 21 values per wheel in [-1, 1] with step 0.1 ⇒ 441 actions
        wheel_vals = np.round(np.linspace(-1.0, 1.0, 21), 2)
        self._lookup = np.stack(np.meshgrid(wheel_vals, wheel_vals), axis=-1).reshape(-1, 2)

        self.action_space = spaces.Discrete(len(self._lookup))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.env.observation_dim,), dtype=np.float32)

        # convenient aliases ------------------------------------------------
        self.observation_dim = self.env.observation_dim
        self.n_actions = len(self._lookup)
        self.discount = self.env.discount

    # ------------------------------------------------ Gym‑style API -------
    def reset(self, **kwargs):
        """Return just the observation (no info dict) so legacy code works."""
        return self.env.reset()


    def step(self, action_idx):
        action = self._lookup[action_idx]
        return self.env.step(action)

    # passthrough helpers ---------------------------------------------------
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.quit_pygame()