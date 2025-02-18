from sample_factory.launcher.launcher_utils import seeds
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("env", ["health_gathering_supreme","battle"]),
        ("seed", range(5)),
        ("num_contexts", [5]),
        ("max_pure_expl_steps", [0,200])
    ]
)

_experiment = Experiment(
    "dc",
    "python -m sf_examples.vizdoom.train_contextual_vizdoom "+
        "--save_every_sec=120 --wandb_project=vizdoom-slurm-feb18"+
        "--with_wandb=True --train_for_env_steps=4000000000 "+
        "--algo=APPO --env_frameskip=4 --use_rnn=True "+
        "--wide_aspect_ratio=False "+
        "--save_milestones_sec=120 --milestone_step_freq=1000000 "+
        "--num_envs_per_worker=2 --decorrelate_envs_on_one_worker=False --num_workers=8",
    _params.generate_params(randomize=False),
)
# --force_envs_single_thread=True

RUN_DESCRIPTION = RunDescription("doom_contexts", experiments=[_experiment])

# --with_wandb=True
# run:
# python -m sample_factory.launcher.run --run=sf_examples.vizdoom.experiments.doom_basic --backend=processes --num_gpus=1 --max_parallel=2  --pause_between=0 --experiments_per_gpu=2
