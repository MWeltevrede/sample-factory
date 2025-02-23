from run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("env", ['my_way_home_1']),
        ("seed", range(0,10)),
        ("num_contexts", [5]),
        ("max_pure_expl_steps", [0, 200])
    ]
)

# _params = ParamGrid(
#     [
#         ("env", ["my_way_home_2"]),
#         ("seed", [25]),
#         ("num_contexts", [5]),
#         ("max_pure_expl_steps", [0])
#     ]
# )

_experiment = Experiment(
    "dc",
    "python -m sf_examples.vizdoom.train_contextual_vizdoom "+
        "--save_every_sec=1200 --wandb_project=vizdoom-slurm "+
        "--with_wandb=True --train_for_env_steps=1000000000 "+
        "--algo=APPO --env_frameskip=4 --use_rnn=True "+
        "--wide_aspect_ratio=False "+
        "--save_milestones_sec=120 --milestone_step_freq=1000000 "+
        "--num_workers=16 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False",
    _params.generate_params(randomize=False),
)
# --force_envs_single_thread=True

RUN_DESCRIPTION = RunDescription("doom_contexts", experiments=[_experiment])

# --with_wandb=True
# run:
# python -m sample_factory.launcher.run --run=sf_examples.vizdoom.experiments.doom_basic --backend=processes --num_gpus=1 --max_parallel=2  --pause_between=0 --experiments_per_gpu=2
