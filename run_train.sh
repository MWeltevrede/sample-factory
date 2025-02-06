# python -m sf_examples.vizdoom.enjoy_vizdoom --save_video --env=doom_battle \
# --experiment=DoomBasic --train_dir=./train_dir --num_workers=1 --num_envs_per_worker=2 \
# --train_for_env_steps=100_000 --with_wandb=True

#python -m sf_examples.vizdoom.enjoy_vizdoom --max_num_frames=1000 --save_video --env=doom_battle --algo=APPO --experiment=DoomBasic --train_dir=./train_dir


#python -m sf_examples.vizdoom.train_contextual_vizdoom --with_wandb=True --algo=APPO --env=doom_battle_contexts --experiment=doom_contexts --save_every_sec=5 --experiment_summaries_interval=10


python -m sample_factory.launcher.run --run=sf_examples.vizdoom.experiments.doom_battle_contexts --backend=processes --num_gpus=1 --max_parallel=4  --pause_between=0 --experiments_per_gpu=2