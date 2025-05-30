import sys
import os
import torch
import pygame
import tqdm
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

flags.DEFINE_string('project_name', 'sac', 'project name')
flags.DEFINE_string('run_name', 'default_run_name', 'name of the run')
flags.DEFINE_string('group_name', 'default_group_name', 'name of the group')
flags.DEFINE_string('layout', '4-1', 'Chosen maze layout.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('max_steps', 500000, 'Number of training steps.')
flags.DEFINE_integer('start_training', 5000,
                     'The step to start training at (build buffer before that).')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('eval_interval', 5000, 'evaluation frequency')
flags.DEFINE_integer('log_interval', 500, 'logging frequency frequency')
flags.DEFINE_integer('eval_episodes', 10, 'num of episodes to evaluate')
flags.DEFINE_boolean('train_render', False,
                     'whether to render during training')
flags.DEFINE_boolean('eval_render', False,
                     'whether to render during evaluation')
flags.DEFINE_boolean('log', True, 'whether to log things to wandb')
flags.DEFINE_integer('partition_size', 10,
                     'size that subdivides a cell of the big maze into minicells')
flags.DEFINE_boolean('use_PER', True, 'to use PER or not')
flags.DEFINE_boolean('manual_control', False,
                     'whether takes actions from manual control or not')
flags.DEFINE_integer(
    'lidar_count', 4, 'the number of lidar angles the robot has access to')
flags.DEFINE_string('save_path', 'models/', 'folder to save models')
flags.DEFINE_string('load_model_path', "", 'path to pre-trained model')
flags.DEFINE_bool('save_model', False, 'to save model or not')

flags.DEFINE_float('h_alpha', 0.3, 'PER alpha')
flags.DEFINE_float('h_beta', 0.4, 'PER beta')

flags.DEFINE_integer('wall_reward_per_layout', -25, 'Reward when hitting wall (negative)')
flags.DEFINE_integer('center_reward_per_layout', 25, 'reward when reaching center of maze (positive)')
flags.DEFINE_integer('max_episodes_per_layout', 100, 'number of episodes per 1 unit of layout')
flags.DEFINE_bool('incremental_training', True, 'whether to gradually step up maximum number of episodes allowed')
flags.DEFINE_bool('incremental_size_training', False, 'whether to gradually place bot further from maze center')
flags.DEFINE_integer('version', 2, 'versioning')
def main(_):
    logging.set_verbosity(logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Run name: {FLAGS.run_name}")
    layout = FLAGS.layout
    size = int(layout.split('-')[0])
    set_seed(FLAGS.seed)
    if layout not in MAZES:
        logging.error(f"Maze layout {layout} is not available in MAZES.")
        return

    rewards_dict = {"wall": FLAGS.wall_reward_per_layout * size, "center": FLAGS.center_reward_per_layout * size}
    env = RatEnv(size, MAZES[layout], max_episode_length=int(FLAGS.max_episodes_per_layout * size), partition_size=FLAGS.partition_size, use_pygame=FLAGS.train_render, rewards=rewards_dict, lidar_count=FLAGS.lidar_count, incremental_training=FLAGS.incremental_training, incremental_size_training=FLAGS.incremental_size_training, max_steps = FLAGS.max_steps)
    eval_env = RatEnv(size, MAZES[layout], max_episode_length=int(FLAGS.max_episodes_per_layout * size), partition_size=FLAGS.partition_size, use_pygame=FLAGS.eval_render, rewards=rewards_dict, lidar_count=FLAGS.lidar_count, incremental_training=False, incremental_size_training=False, max_steps=FLAGS.max_steps)
    
    observation, done = env.reset(0), False
    agent = SACAgent(
        env,
        device,
        batch_size=FLAGS.batch_size,
        critic_kwargs={"hidden_dims": (256, 256, 256)},
        actor_kwargs={"hidden_dims": (256, 256, 256)}
    )

    if (FLAGS.load_model_path != ""):
        print(f"Loading model from: {FLAGS.load_model_path}")
        if not os.path.exists(FLAGS.load_model_path + "/actor.pt"):
            raise FileNotFoundError("No actor model found!")
        if not os.path.exists(FLAGS.load_model_path + "/critic.pt"):
            raise FileNotFoundError("No critic model found!")
        if not os.path.exists(FLAGS.load_model_path + "/target_critic.pt"):
            raise FileNotFoundError("No target critic model found!")
        if not os.path.exists(FLAGS.load_model_path + "/log_alpha.pt"):
            raise FileNotFoundError("No log alpha model found!")

        agent.actor.load_state_dict(torch.load(
            FLAGS.load_model_path + "/actor.pt", map_location=device))
        agent.critic.load_state_dict(torch.load(
            FLAGS.load_model_path + "/critic.pt", map_location=device))
        agent.target_critic.load_state_dict(torch.load(
            FLAGS.load_model_path + "/target_critic.pt", map_location=device))
        agent.log_alpha = torch.load(FLAGS.load_model_path + "/log_alpha.pt", map_location=device)
        agent.log_alpha.requires_grad = True

    evaluate_info = evaluate(
        agent, eval_env, FLAGS.eval_episodes, 0)

if __name__ == "__main__":
    app.run(main)
