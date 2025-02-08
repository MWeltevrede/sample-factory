import functools
import sys

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec, doom_env_by_name
from sf_examples.vizdoom.doom.wrappers.episode_step import EpisodeStepWrapper
from sf_examples.vizdoom.doom.doom_utils import DoomSpec, make_doom_env_from_spec


def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()

def register_custom_doom_env(name='doom_battle_contexts_train'):
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    base_env_spec = doom_env_by_name(name)
    spec = DoomSpec(
        name,
        base_env_spec.env_spec_file,  # use your custom cfg here
        base_env_spec.action_space,
        extra_wrappers=[(EpisodeStepWrapper, {})],
    )

    # register the env with Sample Factory
    make_env_func = functools.partial(make_doom_env_from_spec, spec)
    register_env(spec.name, make_env_func)


def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_vizdoom_components()
    cfg = parse_vizdoom_cfg()

    register_custom_doom_env(name=cfg.env)

    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
