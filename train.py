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
flags.DEFINE_string('save_actor_name', 'best_actor.pt',
                    'path to pretrained actor model')
flags.DEFINE_string('save_critic_name', 'best_critic.pt',
                    'path to pretrained critic model')
flags.DEFINE_boolean('load_model', False,
                     'to train on a pretrained model or not')
flags.DEFINE_string('actor_path', 'models/best_actor.pt',
                    'path to pretrained actor model')
flags.DEFINE_string('critic_path', 'models/best_critic.pt',
                    'path to pretrained critic model')

flags.DEFINE_float('h_alpha', 0.3, 'PER alpha')
flags.DEFINE_float('h_beta', 0.4, 'PER beta')


def main(_):
    logging.set_verbosity(logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    layout = FLAGS.layout
    size = int(layout.split('-')[0])
    set_seed(FLAGS.seed)
    if layout not in MAZES:
        logging.error(f"Maze layout {layout} is not available in MAZES.")
        return

    env = RatEnv(size, MAZES[layout], partition_size=FLAGS.partition_size, use_pygame=FLAGS.train_render)
    eval_env = RatEnv(size, MAZES[layout], max_episode_length=300, use_pygame=FLAGS.eval_render)
    observation, done = env.reset(), False
    agent = SACAgent(
        env,
        device,
        batch_size=FLAGS.batch_size,
        critic_kwargs={"hidden_dims": (256, 256, 256)},
        actor_kwargs={"hidden_dims": (256, 256, 256)}
    )

    if FLAGS.load_model:
        if not os.path.exists(FLAGS.actor_path):
            raise FileNotFoundError(f"No model found at {FLAGS.actor_path}")
        if not os.path.exists(FLAGS.critic_path):
            raise FileNotFoundError(f"No model found at {FLAGS.critic_path}")
        agent.actor.load_state_dict(torch.load(
            FLAGS.actor_path, map_location=device))
        agent.critic.load_state_dict(torch.load(
            FLAGS.critic_path, map_location=device))

    replay_buffer = ReplayBuffer(env.observation_dim, env.action_dim, int(
        FLAGS.max_steps/5), device, FLAGS.use_PER, FLAGS.h_alpha, FLAGS.h_beta, (1.0 - FLAGS.h_beta)/FLAGS.max_steps)

    if FLAGS.log:
        wandb.init(
            entity="ratatouille",
            project="sac",
            group=FLAGS.group_name,
            name=FLAGS.run_name,
        )
        wandb.config.update(FLAGS)

    os.makedirs(FLAGS.save_path, exist_ok=True)
    best_return = -float('inf')

    for i in tqdm.tqdm(range(FLAGS.max_steps), smoothing=0.1):
        if FLAGS.manual_control:
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

        assert (env.runnable)

        logging.debug(f"Step {i}/{FLAGS.max_steps}")
        logging.debug(f"Received observation = {observation}")

        if FLAGS.manual_control:
            action = env.manual_control()
        else:
            action = agent.act(observation)
        logging.debug(f"Action: {action}")
        next_observation, reward, terminal, truncated, info = env.step(action)
        replay_buffer.insert(observation, action, reward,
                             next_observation, terminal)
        observation = next_observation

        done = terminal or truncated
        env.render(f"Step: {i}")
        if done:
            env.clock_tick(10)
        else:
            env.clock_tick(120)

        if done:
            observation, done = env.reset(), False
            if i >= int(3 * FLAGS.max_steps/4):
                env.max_episode_length = 300
            elif i >= int(2 * FLAGS.max_steps/4):
                env.max_episode_length = 250
            elif i >= int(FLAGS.max_steps/4):
                env.max_episode_length = 200

        if i >= FLAGS.start_training:
            observation_batch, action_batch, reward_batch, next_observation_batch, discount_mask_batch, is_weights, idxs, probs = replay_buffer.sample(
                FLAGS.batch_size)
            train_info, td_error_np = agent.update(
                observation_batch, action_batch, reward_batch, next_observation_batch, discount_mask_batch, i, is_weights)

            is_weights_np = to_np(is_weights)

            # kl(p || Unif(N = len(replay_buffer)))
            N = len(replay_buffer)
            mask = probs > (1.0/N)/1000
            kl = np.sum(probs[mask] * (np.log(probs[mask]) + np.log(N)))
            replay_info = {
                "td_max": np.max(td_error_np),
                "td_min": np.min(td_error_np),
                "td_mean": np.mean(td_error_np),
                "td_75_percentile": np.percentile(td_error_np, 75),
                "td_50_percentile": np.percentile(td_error_np, 50),
                "td_25_percentile": np.percentile(td_error_np, 25),
                "is_weights_max": np.max(is_weights_np),
                "is_weights_min": np.min(is_weights_np),
                "is_weights_mean": np.mean(is_weights_np),
                "is_weights_75_percentile": np.percentile(is_weights_np, 75),
                "is_weights_50_percentile": np.percentile(is_weights_np, 50),
                "is_weights_25_percentile": np.percentile(is_weights_np, 25),
                "N * p_max": np.max(probs) * N,
                "N * p_min": np.min(probs) * N,
                "N * p_mean": np.mean(probs) * N,
                "N * p_75_percentile": np.percentile(probs, 75) * N,
                "N * p_50_percentile": np.percentile(probs, 50) * N,
                "N * p_25_percentile": np.percentile(probs, 25) * N,
                "kl": kl,
            }
            # logging.info(replay_info)
            if FLAGS.use_PER:
                replay_buffer.update_priorities(idxs, td_error_np)
            if (FLAGS.log and i % FLAGS.log_interval == 0):
                for k, v in train_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

                for k, v in replay_info.items():
                    wandb.log({f"debug/{k}": v}, step=i)

            if i % FLAGS.eval_interval == 0:
                evaluate_info = evaluate(
                    agent, eval_env, FLAGS.eval_episodes, i)
                average_return = evaluate_info["average_episode_return"]
                if FLAGS.log:
                    for k, v in evaluate_info.items():
                        wandb.log({f"evaluate/{k}": v}, step=i)

                if average_return > best_return:
                    best_return = average_return
                    torch.save(agent.actor.state_dict(), os.path.join(
                        FLAGS.save_path, FLAGS.save_actor_name))
                    torch.save(agent.critic.state_dict(), os.path.join(
                        FLAGS.save_path, FLAGS.save_critic_name))
                    logging.info(
                        f"Saved new best model at step {i} with return {average_return:.2f}")


if __name__ == "__main__":
    app.run(main)
