from typing import Optional

from sample_factory.utils.utils import log

try:
    import envpool
except ImportError as e:
    print(e)
    print("Trying to import envpool when it is not installed. install with 'pip install envpool'")

from sf_examples.envpool.envpool_wrappers import BatchedRecordEpisodeStatistics, EnvPoolResetFixWrapper, EnvPoolNonBatchedWrapper
from sf_examples.vizdoom.doom.wrappers.explore_go import ExploreGoWrapper

class ProcgenSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


ENVPOOL_PROCGEN_ENVS = [
    ProcgenSpec("procgen_bigfish", "BigfishEasy-v0"),
    ProcgenSpec("procgen_bossfight", "BossfightEasy-v0"),
    ProcgenSpec("procgen_caveflyer", "CaveflyerEasy-v0"),
    ProcgenSpec("procgen_chaser", "ChaserEasy-v0"),
    ProcgenSpec("procgen_climber", "ClimberEasy-v0"),
    ProcgenSpec("procgen_coinrun", "CoinrunEasy-v0"),
    ProcgenSpec("procgen_dodgeball", "DodgeballEasy-v0"),
    ProcgenSpec("procgen_fruitbot", "FruitbotEasy-v0"),
    ProcgenSpec("procgen_heist", "HeistEasy-v0"),
    ProcgenSpec("procgen_jumper", "JumperEasy-v0"),
    ProcgenSpec("procgen_leaper", "LeaperEasy-v0"),
    ProcgenSpec("procgen_maze", "MazeEasy-v0"),
    ProcgenSpec("procgen_miner", "MinerEasy-v0"),
    ProcgenSpec("procgen_ninja", "NinjaEasy-v0"),
    ProcgenSpec("procgen_plunder", "PlunderEasy-v0"),
    ProcgenSpec("procgen_starpilot", "StarpilotEasy-v0"),
]


def procgen_env_by_name(name):
    for cfg in ENVPOOL_PROCGEN_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Procgen env")


def make_procgen_env(env_name, cfg, env_config, render_mode: Optional[str] = None, non_batched: Optional[bool] = False):
    if not non_batched and cfg.num_envs_per_worker > 1:
        log.warning(
            "When using envpool, set num_envs_per_worker=1 and use --env_agents={desired number of envs}. "
            f"Setting --num_envs_per_worker={cfg.num_envs_per_worker} will create multiple envpools per worker process "
            f"which is not the desirable behavior in most configurations."
        )
    procgen_spec = procgen_env_by_name(env_name)

    env_kwargs = dict()

    if env_config is not None:
        env_kwargs["seed"] = env_config.env_id

    env = envpool.make(
        procgen_spec.env_id,
        env_type="gym",
        num_envs=cfg.env_agents,
        num_levels=cfg.num_levels,
        start_level=cfg.start_level,
        **env_kwargs,
    )

    env = EnvPoolResetFixWrapper(env)
    env = BatchedRecordEpisodeStatistics(env, num_envs=cfg.env_agents)
    if non_batched:
        env = EnvPoolNonBatchedWrapper(env)
    env = ExploreGoWrapper(env, max_pure_expl_steps=0)
    env.num_agents = cfg.env_agents

    return env
