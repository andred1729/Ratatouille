import torch
import os
import pygame
from ratatouille.env import RatEnv, MAZES
from ratatouille.agents import SACAgent
from ratatouille.utils import set_seed
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string('actor_path', 'models/best_actor.pt', 'path to actor')
flags.DEFINE_string('layout', '4-1', 'Chosen maze layout.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

def load_and_run(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(FLAGS.seed)

    layout = FLAGS.layout
    size = int(layout.split('-')[0])
    if layout not in MAZES:
        raise ValueError(f"Maze size {layout} not found.")
    
    env = RatEnv(size, MAZES[layout], max_episode_length=300, use_pygame=True)
    env.init_pygame()
    observation, done = env.reset(), False

    agent = SACAgent(
        env,
        device,
        actor_kwargs={"hidden_dims": (256, 256, 256)},
        critic_kwargs={"hidden_dims": (256, 256, 256)}
    )

    if not os.path.exists(FLAGS.actor_path):
        raise FileNotFoundError(f"No model found at {FLAGS.actor_path}")
    
    agent.actor.load_state_dict(torch.load(FLAGS.actor_path, map_location=device))
    agent.actor.eval()

    total_reward = 0

    for step in range(300):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
        assert(env.runnable)
        
        action = agent.act(observation)
        observation, reward, terminal, truncated, info = env.step(action)
        total_reward += reward

        env.render(f"Step {step}, Total reward: {total_reward:.2f}")
        pygame.display.update()
        env.clock.tick(15)

        if terminal or truncated:
            break

    print(f"Episode complete. Steps: {step+1}, Total reward: {total_reward:.2f}")
    pygame.quit()

if __name__ == "__main__":
    app.run(load_and_run)

