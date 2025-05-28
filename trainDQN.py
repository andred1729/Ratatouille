import os, sys, tqdm, pygame, torch, numpy as np
from absl import app, flags, logging
import wandb

from ratatouille.env import RatEnv, MAZES
from ratatouille.env.discrete_action_wrapper import DiscreteActionWrapper
from ratatouille.data.replaybufferQ import ReplayBufferDQN
from ratatouille.agents.DQN import DQNAgent
from ratatouille.utils import set_seed
from ratatouille.evaluation import evaluate

FLAGS = flags.FLAGS
# ---------------------------------------------------------------------------
flags.DEFINE_string('run_name', 'dqn_run', 'wandb run name')
flags.DEFINE_string('group_name', 'dqn_group', 'wandb group name')
flags.DEFINE_string('layout', '4-1', 'maze layout (key in MAZES)')
flags.DEFINE_integer('seed', 42, 'random seed')
flags.DEFINE_integer('max_steps', 500000, 'total environment steps')
flags.DEFINE_integer('start_training', 5000, 'steps before learning starts')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('eval_interval', 5000, 'evaluation frequency')
flags.DEFINE_integer('log_interval', 500, 'wandb log frequency')
flags.DEFINE_integer('eval_episodes', 20, 'episodes per evaluation')
flags.DEFINE_boolean('train_render', False, 'render during training')
flags.DEFINE_boolean('eval_render', True, 'render during eval')
flags.DEFINE_boolean('log', True, 'enable wandb')
flags.DEFINE_integer('partition_size', 10, 'maze partition granularity')
flags.DEFINE_string('save_path', 'models/', 'folder to save best model')
flags.DEFINE_string('save_model_name', 'best_dqn.pt', 'filename to save weights')
# ---------------------------------------------------------------------------

def main(_):
    logging.set_verbosity(logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(FLAGS.seed)

    # ---------- Environment setup ----------------------------------------
    layout = FLAGS.layout
    size = int(layout.split('-')[0])
    if layout not in MAZES:
        logging.error(f"Layout {layout} not found in MAZES"); return

    base_env = RatEnv(size, MAZES[layout], partition_size=FLAGS.partition_size,
                      use_pygame=FLAGS.train_render)
    env = DiscreteActionWrapper(base_env)
    eval_env = DiscreteActionWrapper(RatEnv(size, MAZES[layout], max_episode_length=600,
                                            partition_size=FLAGS.partition_size,
                                            use_pygame=FLAGS.eval_render))

    buffer = ReplayBufferDQN(env.observation_dim, capacity=int(FLAGS.max_steps/5), device=device)
    agent = DQNAgent(env, device, buffer, batch_size=FLAGS.batch_size)

    if FLAGS.log:
        wandb.init(project='dqn', group=FLAGS.group_name, name=FLAGS.run_name)
        wandb.config.update(FLAGS)

    os.makedirs(FLAGS.save_path, exist_ok=True)
    best_return = -float('inf')

    obs = env.reset()
    for step in tqdm.tqdm(range(FLAGS.max_steps)):
        # --- keyboard quit option ---------------------------------------
        if FLAGS.train_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        # ---------- acting ----------------------------------------------
        action = agent.act(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        discount = 0.0 if term else env.discount
        buffer.insert(obs, action, reward, next_obs, discount)
        obs = next_obs if not done else env.reset()[0]

        # ---------- learning --------------------------------------------
        if step >= FLAGS.start_training:
            agent.step_count = step  # update internal counter
            log_info = agent.update()
            if FLAGS.log and step % FLAGS.log_interval == 0 and log_info:
                wandb.log(log_info, step=step)

        # ---------- evaluation ------------------------------------------
        if step % FLAGS.eval_interval == 0 and step > 0:
            eval_info = evaluate(agent, eval_env, FLAGS.eval_episodes, step)
            avg_ret = eval_info['average_episode_return']
            if FLAGS.log:
                for k, v in eval_info.items():
                    wandb.log({f'eval/{k}': v}, step=step)
            if avg_ret > best_return:
                best_return = avg_ret
                torch.save(agent.q_net.state_dict(), os.path.join(FLAGS.save_path, FLAGS.save_model_name))
                logging.info(f'New best model saved at step {step} (return={avg_ret:.2f})')

if __name__ == '__main__':
    app.run(main)