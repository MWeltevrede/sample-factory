import gymnasium as gym
import numpy as np


class EpisodeStepWrapper(gym.Wrapper):
    """Doom wrapper to add episode step count to observation."""

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        
    def step(self, action, **kwargs):
        self.step_count += 1
        obs, reward, terminated, truncated, info = super().step(action, **kwargs)
        info['step_count'] = self.step_count
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
        self.step_count = 0
        obs, info =  self.env.reset(**kwargs)
        info['step_count'] = self.step_count
        return obs, info