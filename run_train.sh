# python -m sample_factory.launcher.run --run=sf_examples.vizdoom.experiments.doom_contexts --backend=processes --num_gpus=1 --max_parallel=1 --pause_between=0 --experiments_per_gpu=2

python -m sample_factory.launcher.run --run=sf_examples.envpool.procgen.experiments.procgen_envpool --backend=processes --num_gpus=1 --max_parallel=1 --pause_between=0 --experiments_per_gpu=1
# python -m sf_examples.envpool.procgen.experiments.procgen_envpool

# python -m sample_factory.launcher.run --run=sf_examples.envpool.mujoco.experiments.mujoco_envpool --backend=processes --num_gpus=1 --max_parallel=1 --pause_between=0 --experiments_per_gpu=1