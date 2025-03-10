import sys
from typing import Optional

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.envpool.procgen.envpool_procgen_params import add_procgen_env_args, procgen_override_defaults
from sf_examples.envpool.procgen.envpool_procgen_utils import ENVPOOL_PROCGEN_ENVS, make_procgen_env

def non_batched_make_procgen_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    return make_procgen_env(env_name, cfg, env_config, render_mode, non_batched=True)

def register_procgen_envs():
    for env in ENVPOOL_PROCGEN_ENVS:
        register_env(env.name, non_batched_make_procgen_env)


def register_procgen_components():
    register_procgen_envs()


def parse_procgen_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_procgen_env_args(partial_cfg.env, parser, evaluation=evaluation)
    procgen_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_procgen_components()
    cfg = parse_procgen_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
