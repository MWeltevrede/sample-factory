from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sample_factory.utils.algo_version import ALGO_VERSION

_params = ParamGrid(
    [
        ("seed", [1]),
        (
            "env",
            [
                "procgen_bigfish",
            ],
        ),
    ]
)

vstr = f"procgen_envpool"
cli = (
    f"python -m sf_examples.envpool.procgen.train_envpool_procgen "
    f"--train_for_env_steps=25000000 --with_wandb=True --wandb_tags {vstr} --wandb_group=sf2_{vstr} --num_workers=16 --num_envs_per_worker=16 --rollout=64 --batched_sampling=False --env_agents=1 --decorrelate_experience_max_seconds=0 --decorrelate_envs_on_one_worker=False"
)

_experiments = [Experiment(f"{vstr}", cli, _params.generate_params(False))]
RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)

# Run locally: python -m sample_factory.launcher.run --run=sf_examples.envpool.mujoco.experiments.mujoco_envpool --backend=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.launcher.run --run=sf_examples.envpool.mujoco.experiments.mujoco_envpool --backend=slurm --slurm_workdir=./slurm_envpool --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False

