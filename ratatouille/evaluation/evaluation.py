from typing import Dict
import numpy as np
from ratatouille.env import RatEnv
import imageio
import pygame
def evaluate(agent, env: RatEnv, num_episodes, step, save_video = False) -> Dict[str, float]:
    if save_video:
        # save the video here
        pass
    
    sum_episode_return = 0.0
    sum_episode_length = 0.0
    n_wins = 0
    
    env.init_pygame()
    
    frames = []
    for i in range(num_episodes):
        observation, done = env.reset(0), False
        info = {}
        
        frames.clear()
        rendered = False
        while not done:
            env.render(f"Current Step: {step}. Evaluating Episode {i+1}/{num_episodes}.")
            if not rendered:
                frame = pygame.surfarray.array3d(pygame.display.get_surface())
                frame = np.transpose(frame, (1, 0, 2))
                imageio.imwrite(f"videos/init_frame_ep{i+1}.png", frame)
                rendered = True
        
            env.clock_tick(30)
            if save_video:
                frame = pygame.surfarray.array3d(pygame.display.get_surface())
                frame = np.transpose(frame, (1, 0, 2))
                frames.append(frame)
            action = agent.act(observation)
            next_observation, reward, terminal, truncated, info = env.step(action)

            observation = next_observation
            done = terminal or truncated
        env.clock_tick(5)
        
        sum_episode_length += info["current_episode_length"]
        sum_episode_return += info["current_episode_discounted_return"]
        n_wins += int(info["is_win"])

        if save_video:
            video_path = f"videos/eval_ep_{i+1}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            print(f"saved video: {video_path}")
    env.quit_pygame()


    return {
        "average_episode_length": sum_episode_length/num_episodes,
        "average_episode_return":  sum_episode_return/num_episodes,
        "completion_rate": float(n_wins)/num_episodes
    }