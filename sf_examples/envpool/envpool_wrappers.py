from typing import Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

DoneStepType = Tuple[
    Union[ObsType, np.ndarray],
    Union[float, np.ndarray],
    Union[bool, np.ndarray],
    Union[dict, list],
]

TerminatedTruncatedStepType = Tuple[
    Union[ObsType, np.ndarray],
    Union[float, np.ndarray],
    Union[bool, np.ndarray],
    Union[bool, np.ndarray],
    Union[dict, list],
]


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2


class EnvPoolResetFixWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions, *args):
        obs, reward, terminated, truncated, info = self.env.step(actions, *args)

        needs_reset = np.nonzero(terminated | truncated)[0]
        obs[needs_reset], _ = self.env.reset(needs_reset)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        kwargs.pop("seed", None)  # envpool does not support the seed in reset, even with the updated API
        kwargs.pop("options", None)
        return self.env.reset(**kwargs)
    
class EnvPoolNonBatchedWrapper(gym.Wrapper):
    """
        This wrapper is necessary when running Explore-Go. 
        It runs envpool in non-batched (or non-vectorised) mode, which is less efficient but necessary for some algorithms.
    """
    def __init__(self, env):
        super().__init__(env)
        unwrapped_env = env
        while hasattr(unwrapped_env, "env"):
            unwrapped_env = unwrapped_env.env
        self.action_dtype = unwrapped_env.spec.action_array_spec['action'].dtype
        self.env_id_dtype = unwrapped_env.spec.action_array_spec['env_id'].dtype

    def step(self, actions):
        actions = np.array([actions], dtype=self.action_dtype)
        env_ids = np.array([0], dtype=self.env_id_dtype)

        obs, reward, terminated, truncated, info = self.env.step(actions, env_ids)

        obs = obs[0]
        reward = reward[0]
        terminated = terminated[0]
        truncated = truncated[0]
        for k,v in info.items():
            if isinstance(v, dict):
                for k2,v2 in v.items():
                    v[k2] = v2[0]
            else:
                info[k] = v[0]

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs[0]
        for k,v in info.items():
            if isinstance(v, dict):
                for k2,v2 in v.items():
                    v[k2] = v2[0]
            else:
                info[k] = v[0]
        return obs, info


class BatchedRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, num_envs, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", num_envs)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action, *args):
        observations, rewards, terminated, truncated, infos = self.env.step(action, *args)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - terminated
        self.episode_lengths *= 1 - terminated
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, terminated, truncated, infos
