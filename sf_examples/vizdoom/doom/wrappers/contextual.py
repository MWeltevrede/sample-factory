import gymnasium as gym
import numpy as np


class ContextualWrapper(gym.Wrapper):
    """Doom wrapper to change screen resolution."""

    def __init__(self, env, num_contexts):
        super().__init__(env)
        self.num_contexts = num_contexts
        self.seeds = range(num_contexts)
        
    def step(self, action, **kwargs):
        if self.num_contexts > 0:
            assert self.game.get_seed() == int(self.cur_seed)
        return super().step(action, **kwargs)

    def reset(self, **kwargs):
        if self.num_contexts > 0:
            self.cur_seed = np.random.choice(self.seeds)
            kwargs['seed'] = int(self.cur_seed)
            out = self.env.reset(**kwargs)
            self.game.set_seed(self.cur_seed)
            assert self.game.get_seed() == int(self.cur_seed)
            return out
        else:
            return self.env.reset(**kwargs)