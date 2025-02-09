"""
Example of how to use VizDoom env API to use your own custom VizDoom environment with Sample Factory.

To train:
python -m sf_examples.vizdoom.train_custom_vizdoom_env --algo=APPO --env=doom_my_custom_env --experiment=doom_my_custom_env_example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sf_examples.vizdoom.enjoy_custom_vizdoom_env --algo=APPO --env=doom_my_custom_env --experiment=doom_my_custom_env_example

"""
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

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults, add_doom_env_eval_args
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

from sf_examples.vizdoom.train_contextual_vizdoom import register_custom_doom_env
        
def add_custom_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num_contexts", type=int, default=0, help="The number of contexts (seeds) seen during training.")

def main():
    """Script entry point."""
    register_vizdoom_components()

    parser, cfg = parse_sf_args(evaluation=True)
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    doom_override_defaults(parser)
    add_custom_args(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)

    register_custom_doom_env(name='doom_battle_contexts', num_contexts=cfg.num_contexts, max_pure_expl_steps=cfg.max_pure_expl_steps)
    register_custom_doom_env(name='doom_battle_contexts_test', num_contexts=-1)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
