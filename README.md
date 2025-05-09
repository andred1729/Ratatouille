# Environment specs
- Robot starts at bottom left corner
- Discrete time unit is $dt = 0.1$
- Acceleration should be within $[-1, 1]$
- Max speed is $1$, and cell size is also $1$.
- That roughly means that it takes 10 time steps of full throttle acceration to get top speed.

# Setting up the virtual env
- Make a virtual environment with python >= 3.10, say `.venv`
- Run `pip install requirements.txt`

# Run commands
Run `python train.py` in the project folder.

Optional argument:
- `size` (default = 4): size of the maze

e.g. `python train.py --size=8`