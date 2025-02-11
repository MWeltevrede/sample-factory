import gymnasium as gym
import numpy as np


class ExploreGoWrapper(gym.Wrapper):
    """Doom wrapper to add episode step count to observation."""

    def __init__(self, env, max_pure_expl_steps):
        super().__init__(env)
        self.step_count = 0
        self.max_pure_expl_steps = max_pure_expl_steps
        self.num_pure_expl_steps = np.random.randint(0, self.max_pure_expl_steps+1)
        
    def step(self, action, **kwargs):
        self.step_count += 1
        obs, reward, terminated, truncated, info = super().step(action, **kwargs)
        info['step_count'] = np.array([np.array(self.step_count, dtype=np.int64), self.num_pure_expl_steps])
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
        self.step_count = 0
        self.num_pure_expl_steps = np.random.randint(0, self.max_pure_expl_steps+1)
        obs, info =  self.env.reset(**kwargs)
        info['step_count'] = np.array([np.array(self.step_count, dtype=np.int64), self.num_pure_expl_steps])
        return obs, info