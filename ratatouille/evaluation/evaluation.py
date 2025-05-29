from typing import Dict
import numpy as np
from ratatouille.env import RatEnv
def evaluate(agent, env: RatEnv, num_episodes, step, save_video = False) -> Dict[str, float]:
    if save_video:
        # save the video here
        pass
    
    sum_episode_return = 0.0
    sum_episode_length = 0.0
    n_wins = 0
    
    env.init_pygame()
    
    for i in range(num_episodes):
        observation, done = env.reset(0), False
        info = {}
        while not done:
            env.render(f"Current Step: {step}. Evaluating Episode {i+1}/{num_episodes}.")
            env.clock_tick(30)
            action = agent.act(observation)
            next_observation, reward, terminal, truncated, info = env.step(action)

            observation = next_observation
            done = terminal or truncated
        env.clock_tick(5)
        
        sum_episode_length += info["current_episode_length"]
        sum_episode_return += info["current_episode_discounted_return"]
        n_wins += int(info["is_win"])
    
    env.quit_pygame()


    return {
        "average_episode_length": sum_episode_length/num_episodes,
        "average_episode_return":  sum_episode_return/num_episodes,
        "completion_rate": float(n_wins)/num_episodes
    }