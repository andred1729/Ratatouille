from typing import Dict
import numpy as np
def evaluate(agent, env, num_episodes, step, save_video = False) -> Dict[str, float]:
    if save_video:
        # save the video here
        pass
    
    sum_episode_return = 0.0
    sum_episode_length = 0.0
    n_wins = 0

    for i in range(num_episodes):
        observation, done = env.reset(), False
        info = {}
        while not done:
            # env.render(f"Current Step: {step}. Evaluating Episode {i+1}/{num_episodes}.")
            # env.clock.tick(120)
            action = agent.act(observation)
            next_observation, reward, terminal, truncated, info = env.step(action)

            observation = next_observation
            done = terminal or truncated
        # env.clock.tick(5)
        
        sum_episode_length += info["current_episode_length"]
        sum_episode_return += info["current_episode_discounted_return"]
        n_wins += int(info["is_win"])
    
    # env.render(f"IN TRAINING. Last evaluated at step {step}.")


    return {
        "average_episode_length": sum_episode_length/num_episodes,
        "average_episode_return":  sum_episode_return/num_episodes,
        "completion_rate": float(n_wins)/num_episodes
    }