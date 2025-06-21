import random
import gym
import numpy as np
from process_image import process_image

class PongWrapper(object):
    def __init__(self, env_name: str, history_length: int = 4, render_mode: str = "rgb_array"):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.render_mode = render_mode
        self.history_length = history_length
        self.state = None

    def reset(self):
        self.frame, _ = self.env.reset()
        self.state = np.repeat(process_image(self.frame), self.history_length, axis=2)
        return self.state

    def step(self, action: int, render_mode=None):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        processed_image = process_image(obs)
        self.state = np.append(self.state[:, :, 1:], processed_image, axis=2)

        if render_mode == 'rgb_array':
            return processed_image, reward, done, self.env.render()
        elif render_mode == 'human':
            self.env.render()

        return processed_image, reward, done


