import sys
import torch
import pygame
from absl import app, flags, logging
import wandb
import numpy as np
from ratatouille.env import RatEnv, MAZES
from ratatouille.utils import set_seed
from ratatouille.data import ReplayBuffer
from ratatouille.agents import SACAgent
from ratatouille.evaluation import evaluate
from ratatouille.utils import to_np
FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 4, 'Size of the maze.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('max_steps', 500000, 'Number of training steps.')
flags.DEFINE_integer('start_training', 1000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('eval_interval', 1000, 'evaluation frequency')
flags.DEFINE_integer('eval_episodes', 10, 'num of episodes to evaluate')
flags.DEFINE_boolean('render', False, 'whether to render')
flags.DEFINE_string('run_name', 'default', 'name of the run')
flags.DEFINE_boolean('log', True, 'whether to log things to wandb')
flags.DEFINE_integer('partition_size', 10, 'size that subdivides a cell of the big maze into minicells')
flags.DEFINE_boolean('use_PER', True, 'to use PER or not')
def main(_):
    logging.set_verbosity(logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    size = FLAGS.size
    set_seed(FLAGS.seed)
    if size not in MAZES:
        logging.error(f"Maze size {size} is not available in MAZES.")
        return
    
    env = RatEnv(size, MAZES[size], FLAGS.partition_size)
    eval_env = RatEnv(size, MAZES[size])
    observation, done = env.reset(), False
    agent = SACAgent(
        env,
        device,
        batch_size=FLAGS.batch_size
    )
    replay_buffer = ReplayBuffer(env.observation_dim, env.action_dim, FLAGS.max_steps, device, FLAGS.use_PER)
    
    if FLAGS.log:
        wandb.init(
            entity="ratatouille",
            project="sac",
        )
        wandb.config.update(FLAGS)
    
    for i in range(FLAGS.max_steps):
        if FLAGS.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                # Handle keyboard events
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        env.reset()
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
        
        assert(env.runnable)
        
        logging.info(f"Step {i}/{FLAGS.max_steps}")
        logging.debug(f"Received observation = {observation}")
        # action = env.manual_control()
        # action = np.tanh(np.random.uniform(-1, 1, env.action_dim))
        action = agent.act(observation)
        logging.debug(f"Action: {action}")
        next_observation, reward, terminal, truncated, info = env.step(action)
        replay_buffer.insert(observation, action, reward, next_observation, terminal)
        observation = next_observation
        
        done = terminal or truncated
        if FLAGS.render:
            env.render(f"Step: {i}")
            if done:
                env.clock.tick(1)
            else:
                env.clock.tick(60)
        
        if done:
            observation, done = env.reset(), False

        
        if i >= FLAGS.start_training:
            observation_batch, action_batch, reward_batch, next_observation_batch, discount_mask_batch, is_weights, idxs = replay_buffer.sample(FLAGS.batch_size)
            train_info, td_error_np = agent.update(observation_batch, action_batch, reward_batch, next_observation_batch, discount_mask_batch, i, is_weights)
        

            is_weights_np = to_np(is_weights)
            replay_info = {
                "td_max": np.max(td_error_np),
                "td_mean": np.mean(td_error_np),
                "td_75_percentile": np.percentile(td_error_np, 75),
                "td_50_percentile": np.percentile(td_error_np, 50),
                "td_25_percentile": np.percentile(td_error_np, 25),
                "is_weights_max": np.max(is_weights_np),
                "is_weights_mean": np.mean(is_weights_np),
                "is_weights_75_percentile": np.percentile(is_weights_np, 75),
                "is_weights_50_percentile": np.percentile(is_weights_np, 50),
                "is_weights_25_percentile": np.percentile(is_weights_np, 25),
            }
            # logging.info(replay_info)
            if FLAGS.use_PER:
                replay_buffer.update_priorities(idxs, td_error_np)
            if FLAGS.log:
                for k, v in train_info.items():
                    wandb.log({f"training/{k}": v}, step=i)
                
                for k, v in replay_info.items():
                    wandb.log({f"training/{k}": v}, step=i)
            
            if i % FLAGS.eval_interval == 0:
                evaluate_info = evaluate(agent, eval_env, FLAGS.eval_episodes)
                if FLAGS.log:
                    for k, v in evaluate_info.items():
                        wandb.log({f"evaluate/{k}": v}, step=i)

if __name__ == "__main__":
    app.run(main)