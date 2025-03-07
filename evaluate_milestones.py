import imageio
import numpy as np
from pyvirtualdisplay import Display  
import gymnasium as gym
import torch
from os.path import join
import os
import functools
import argparse
import time
import json
import copy

from sample_factory.algo.utils.make_env import make_env_func_non_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sf_examples.vizdoom.train_contextual_vizdoom import register_custom_doom_env
from sample_factory.enjoy import make_env
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.learning.learner import Learner

import pandas as pd 
import wandb
api = wandb.Api()


def init_env(original_cfg, train=True):
    cfg = copy.deepcopy(original_cfg)
    cfg.max_num_frames = 1_000_000
    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"

    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip

    cfg.num_envs = 1
    if train == False:
        cfg.env = cfg.env[:-5]+'test'

    render_mode = "human"

    env = make_env(cfg, render_mode=render_mode)
    env_info = extract_env_info(env, cfg)
    
    return env, env_info

def add_custom_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num_contexts", type=int, default=0, help="The number of contexts (seeds) seen during training.")
    # parser.add_argument("--max_num_episodes", type=int, default=10, help="The number of episodes to evaluate the policy on.")


num_episodes = 100
root_dir = ''
job_ids = []

register_vizdoom_components()
parser, cfg = parse_sf_args(evaluation=True)
add_doom_env_args(parser)
doom_override_defaults(parser)
add_custom_args(parser)
# second parsing pass yields the final configuration
cfg = parse_full_cfg(parser)

cfg.eval_env_frameskip = 4

register_custom_doom_env(name=cfg.env, num_contexts=10)
register_custom_doom_env(name=cfg.env, test=True, num_contexts=-1)

env, env_info = init_env(cfg, train=True)

actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
actor_critic.eval()
device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
actor_critic.model_to_device(device)


for filename in os.scandir(root_dir):
    if filename.name[:2] not in job_ids:
        continue
    if os.path.isfile(root_dir + filename.name + '/evaluation.csv'):
        os.remove(root_dir + filename.name + '/evaluation.csv')
        print(f"Deleting evaluation.csv for {filename.name} because it already exists")
    t0 = time.time()
    checkpoint_dir = root_dir + filename.name + '/checkpoint_p0/milestones'
    with open(root_dir + filename.name + '/config.json', 'r') as file:
        run_cfg = json.load(file)
        run_unique_id = run_cfg['wandb_unique_id']
        max_pure_expl_steps = run_cfg['max_pure_expl_steps']
    run = api.run("max-phd/vizdoom-slurm/" + run_unique_id)
    data = run.history(keys=['0/global_step', '0/eval/total_steps'])
    global_steps = data['0/global_step'].to_numpy()
    global_steps.astype(np.int64)
    total_steps = data['0/eval/total_steps'].to_numpy()
    total_steps.astype(np.int64)
    if np.all(global_steps > total_steps):
        tmp = global_steps
        global_steps = total_steps
        total_steps = tmp
    else:
        assert np.all(global_steps < total_steps), "global_steps is not consistently larger or smaller than total_steps"
    total_to_global_df = pd.DataFrame({'global_steps': global_steps, 'total_steps': total_steps})

    # compute which checkpoints to evaluate
    all_global_step_cps = []
    checkpoints = Learner.get_checkpoints(checkpoint_dir, "*")
    for cp in checkpoints:
        checkpoint_dict = torch.load(cp, map_location='cpu', mmap=True)
        all_global_step_cps.append(checkpoint_dict['env_steps'])
    all_global_step_cps_df = pd.DataFrame({'global_steps': all_global_step_cps})

    cps_to_evaluate = dict()
    for ts in range(500_000_000//20, 500_000_001, 500_000_000//20):
        gs = total_to_global_df.loc[(total_to_global_df.total_steps - ts).abs().idxmin()]['global_steps']
        gs_cp = all_global_step_cps_df.loc[(all_global_step_cps_df.global_steps - gs).abs().idxmin()]['global_steps']
        cps_to_evaluate[gs_cp] = ts

    step_data = []
    performance = []
    data_type = []
    method = []
    checkpoints = Learner.get_checkpoints(checkpoint_dir, "*")
    for cp in checkpoints:
        checkpoint_dict = Learner.load_checkpoint([cp], device)
        if checkpoint_dict['env_steps'] in cps_to_evaluate.keys():
            actor_critic.load_state_dict(checkpoint_dict['model'])

            for evaluate_train in [True, False]:
                episode_rewards = []
                env, env_info = init_env(cfg, train=evaluate_train)
                obs, infos = env.reset()
                for i in range(num_episodes):
                    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
                    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
                    dones = False
                    ep_reward = 0
                    while not dones:
                        # retrieve your action here
                        # action = train_env.action_space.sample() # for example, just sample a random action
                        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                        policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
                        # sample actions from the distribution by default
                        actions = policy_outputs["actions"]

                        # actions shape should be [num_agents, num_actions] even if it's [1, 1]
                        if actions.ndim == 1:
                            actions = unsqueeze_tensor(actions, dim=-1)
                        actions = preprocess_actions(env_info, actions)
                        rnn_states = policy_outputs["new_rnn_states"]

                        obs, reward, terminated, truncated, info = env.step(actions)
                        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
                        dones = make_dones(terminated, truncated)
                        infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos
                        ep_reward += reward
                    episode_rewards.append(ep_reward.item())

                performance.append(np.array(episode_rewards).mean())
                if evaluate_train:
                    data_type.append('Train')
                else:
                    data_type.append('Test')
                step_data.append(cps_to_evaluate[checkpoint_dict['env_steps']])
                method.append(max_pure_expl_steps)
                print("100 episodes evaluated!")
    
    data = pd.DataFrame({'performance': performance, 'steps': step_data, 'type': data_type, 'method':method})
    data.to_csv(root_dir + filename.name + '/evaluation.csv')
    print(f"Evaluating one seed in: {time.time() - t0}s")


                