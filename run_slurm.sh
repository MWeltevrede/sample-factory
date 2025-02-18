python -m sample_factory.launcher.run \
     --run=sf_examples.vizdoom.experiments.doom_contexts \
     --backend=slurm_explorego \
     --slurm_workdir=./slurm_vizdoom \
     --experiment_suffix=slurm \
     --slurm_gpus_per_job=1 \
     --slurm_cpus_per_gpu=16 \
     --pause_between=1 \
     --slurm_print_only=False

#--slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_explorego.sh\