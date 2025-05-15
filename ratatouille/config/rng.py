import numpy as np

rng: np.random.Generator = None

def set_seed(seed):
    global rng
    rng = np.random.default_rng(seed)