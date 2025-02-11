from sample_factory.launcher.launcher_utils import seeds
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
import argparse

_params = ParamGrid(
    [   
        ("env", ['take_cover']),
        ("seed", range(2)),
        ("num_contexts", [20]),
    ]
)
#doom_health_gathering_supreme
#doom_battle_contexts
_experiment = Experiment(
    "dc",
    f"python -m sf_examples.vizdoom.train_contextual_vizdoom\
        --save_every_sec=120 --wandb_project=vizdoom-slurm\
        --with_wandb=True --train_for_env_steps=4000000000\
        --algo=APPO --env_frameskip=4 --use_rnn=True\
        --wide_aspect_ratio=False --num_workers=32\
        --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False",
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('vizdoom_contexts', experiments=[_experiment])

# --with_wandb=True
# run:
# python -m sample_factory.launcher.run --run=sf_examples.vizdoom.experiments.doom_basic --backend=processes --num_gpus=1 --max_parallel=2  --pause_between=0 --experiments_per_gpu=2
