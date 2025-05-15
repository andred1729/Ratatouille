#     level=logging.DEBUG, 
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     datefmt='%H:%M:%S',
#     force=True
# )

from absl import app, flags, logging
from ratatouille.env import RatEnv, MAZES

FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 4, 'Size of the maze')


def main(_):
    logging.set_verbosity(logging.DEBUG)
    size = FLAGS.size
    if size not in MAZES:
        logging.error(f"Maze size {size} is not available in MAZES.")
        return
    
    env = RatEnv(size, MAZES[size])
    
    # Instead of running a fixed loop with constant input,
    # use the built-in run method for keyboard control
    env.run()

if __name__ == "__main__":
    app.run(main)