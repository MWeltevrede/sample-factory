"""
Example of how to use VizDoom env API to use your own custom VizDoom environment with Sample Factory.

To train:
python -m sf_examples.vizdoom.train_custom_vizdoom_env --algo=APPO --env=doom_my_custom_env --experiment=doom_my_custom_env_example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sf_examples.vizdoom.enjoy_custom_vizdoom_env --algo=APPO --env=doom_my_custom_env --experiment=doom_my_custom_env_example

"""
from tensorboardX import SummaryWriter

import time
from collections import deque
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import numpy as np
import argparse
import functools
import os
import sys
from os.path import join

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
# from sample_factory.train import run_rl
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.train import make_runner
from sf_examples.vizdoom.doom.action_space import doom_action_space_extended
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DoomSpec, make_doom_env_from_spec, doom_env_by_name
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components
from sf_examples.vizdoom.doom.wrappers.contextual import ContextualWrapper
from sf_examples.vizdoom.doom.wrappers.explore_go import ExploreGoWrapper
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import BatchedVecEnv, make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log

from sample_factory.enjoy import render_frame, visualize_policy_inputs, load_state_dict, make_env


def evaluate_full_contexts(runner: Runner, writer, env_steps, policy_id) -> None:

    cfg = runner.cfg
    #env_steps = cfg.evaluation_env_steps
    verbose = True

    cfg = load_from_checkpoint(cfg)
    cfg.max_num_frames = 20000

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1
    cfg.env = cfg.env+'_test'

    render_mode = "human"
    # if cfg.save_video:
    #     render_mode = "rgb_array"
    # elif cfg.no_render:
    #     render_mode = None

    env = make_env(cfg, render_mode=render_mode)
    env_info = extract_env_info(env, cfg)

    # if hasattr(env.unwrapped, "reset_on_init"):
    #     # reset call ruins the demo recording for VizDoom
    #     env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    cfg.policy_index = policy_id
    try:
        load_state_dict(cfg, actor_critic, device)
    except:
        return

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []

    obs, infos = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = []
    num_episodes = 0

    with torch.no_grad():
        ep_rews = []
        ep_true_objs = []
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            # if not cfg.no_render:
            #     visualize_policy_inputs(normalized_obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            # if cfg.eval_deterministic:
            #     action_distribution = actor_critic.action_distribution()
            #     actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                # last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                obs, rew, terminated, truncated, infos = env.step(actions)
                action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                # if num_frames % 1000 == 0:
                #     log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective)

                        # if verbose:
                        #     log.info(
                        #         "Evaluation episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                        #         agent_i,
                        #         num_frames,
                        #         episode_reward[agent_i],
                        #         true_objectives[agent_i][-1],
                        #     )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                        if cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i].keys():
                                num_episodes += 1
                                if num_episodes % 10 == 0:
                                    log.debug(f"Num eval episodes {num_episodes}...")
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1
                            reward_list.append(true_objective)

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    # render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_obj = np.mean(true_objectives[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                        if not np.isnan(avg_true_obj):
                            if avg_true_objective_str:
                                avg_true_objective_str += ", "
                            avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                    log.info(
                        "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                    )
                    log.info(
                        "Avg episode reward: %.3f, avg true_objective: %.3f",
                        np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                        np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                    )
                    # use 000 here to put these summaries on top in tensorboard (it sorts by ASCII)
                    ep_rews.append(np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]))
                    ep_true_objs.append(np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]))
               
            if num_episodes >= cfg.max_num_episodes:
                writer.add_scalar("eval/episode_rewards", float(np.mean(ep_rews)), env_steps)
                writer.add_scalar("eval/true_objectives", float(np.mean(ep_true_objs)), env_steps)
                break

    env.close()

    # return ExperimentStatus.SUCCESS, sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / sum(
    #     [len(episode_rewards[i]) for i in range(env.num_agents)]
    # )
    
    
class VizdoomContextualEvaluation(AlgoObserver):
    # def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
    def extra_summaries(self, runner: Runner, policy_id: PolicyID, writer: SummaryWriter, env_steps: int) -> None:
        """Called after each training step."""
        evaluate_full_contexts(runner=runner, writer=writer, env_steps=env_steps, policy_id=policy_id)
        

def add_custom_args(parser: argparse.ArgumentParser) -> None:
    #parser.add_argument("--env", type=str, default='doom_battle_contexts', help="The name of the environment")
    parser.add_argument("--num_contexts", type=int, default=0, help="The number of contexts (seeds) seen during training.")
    parser.add_argument("--max_num_episodes", type=int, default=30, help="The number of episodes to evaluate the policy on.")
    
def register_msg_handlers(cfg: Config, runner: Runner):
    # extra functions to evaluate on the full set of seeds
    runner.register_observer(VizdoomContextualEvaluation())
        
def register_custom_doom_env(base_name='doom_battle', name='doom_battle', num_contexts=-1,  max_pure_expl_steps=0):
def register_custom_doom_env(num_contexts, name, test=False):
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    base_env_spec = doom_env_by_name(base_name)
    scenario_absolute_path = join(os.path.dirname(__file__), "doom", "scenarios", f"{name}.cfg")
    if test:
        name += '_test'
    spec = DoomSpec(
        name,
        base_env_spec.env_spec_file,  # use your custom cfg here
        base_env_spec.action_space,
        extra_wrappers=[(ExploreGoWrapper, {'max_pure_expl_steps': max_pure_expl_steps}), (ContextualWrapper, {'num_contexts': num_contexts})],
    )

    # register the env with Sample Factory
    make_env_func = functools.partial(make_doom_env_from_spec, spec)
    register_env(spec.name, make_env_func)


def main():
    """Script entry point."""
    register_vizdoom_components()

    parser, cfg = parse_sf_args()
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    add_custom_args(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)

    register_custom_doom_env(base_name=cfg.env, name=cfg.env, num_contexts=cfg.num_contexts, max_pure_expl_steps=cfg.max_pure_expl_steps)
    register_custom_doom_env(base_name=cfg.env, name=cfg.env + '_test', num_contexts=-1)

    # ensure there is no env decorrelation since that is basically similar to what Explore-Go is trying to do
    cfg.decorrelate_experience_max_seconds = 0
    cfg.decorrelate_envs_on_one_worker = False
    register_custom_doom_env(name=cfg.env, num_contexts=cfg.num_contexts)
    register_custom_doom_env(name=cfg.env, num_contexts=-1, test=True)

    # explicitly create the runner instead of simply calling run_rl()
    # this allows us to register additional message handlers
    cfg, runner = make_runner(cfg)
    register_msg_handlers(cfg, runner)
    
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status

    #status = run_rl(cfg)
    #return status


if __name__ == "__main__":
    sys.exit(main())
