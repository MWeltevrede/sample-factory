import argparse


def procgen_override_defaults(_env, parser):
    """RL params specific to Atari envs."""
    parser.set_defaults(
        # let's set this to True by default so it's consistent with how we report results for other envs
        # (i.e. VizDoom or DMLab). When running evaluations for reports or to compare with other frameworks we can
        # set this to false in command line
        summaries_use_frameskip=False,
        use_record_episode_statistics=True,
        encoder_conv_architecture="resnet_impala",
        encoder_conv_mlp_layers=[256],
        obs_scale=255.0,
        gamma=0.999,
        env_frameskip=1,
        env_framestack=1,
        exploration_loss_coeff=0.01,
        num_workers=8,
        num_envs_per_worker=32,
        worker_num_splits=1,
        env_agents=1,
        train_for_env_steps=25_000_000,
        nonlinearity="relu",
        kl_loss_coeff=0.0,
        use_rnn=False,
        adaptive_stddev=False,
        reward_scale=1.0,
        with_vtrace=False,
        recurrence=1,
        batch_size=2_048,
        rollout=64,
        max_grad_norm=0.5,
        num_epochs=3,
        num_batches_per_epoch=8,
        ppo_clip_ratio=0.2,
        value_loss_coeff=0.5,
        exploration_loss="entropy",
        learning_rate=5.0e-4,
        lr_schedule="linear_decay",
        shuffle_minibatches=False,
        gae_lambda=0.95,
        batched_sampling=False,
        normalize_input=False,
        normalize_returns=True,
        serial_mode=False,
        async_rl=True,
        experiment_summaries_interval=10,
        adam_eps=1e-5,  # choosing the same value as CleanRL used
        wandb_project="sf_procgen",
        decorrelate_experience_max_seconds=0,
        decorrelate_envs_on_one_worker=False,
    )


def add_procgen_env_args(_env, p: argparse.ArgumentParser, evaluation=False):
    if evaluation:
        # apparently env.render(mode="human") is not supported anymore and we need to specify the render mode in
        # the env ctor
        p.add_argument("--render_mode", default="human", type=str, help="")

    p.add_argument(
        "--env_agents",
        default=2,
        type=int,
        help="Num agents in each envpool (if used)",
    )
    p.add_argument(
        "--num_levels",
        default=200,
        type=int,
        help="Num of training levels",
    )
    p.add_argument(
        "--start_level",
        default=0,
        type=int,
        help="Level ID to start on (will train on the levels in [start_level, start_level+num_levels-1]).",
    )
