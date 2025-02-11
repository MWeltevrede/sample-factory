#!/bin/bash

#SBATCH --job-name=vizdoom
#SBATCH --time 12:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu-a100
# originally 12GB
#SBATCH --mem-per-cpu=6GB
#SBATCH --account=Research-EEMCS-INSY
#SBATCH --output=slurm_out/slurm-%x-%j.out
#SBATCH --error=slurm_out/slurm-%x-%j.err

# modules:
module load 2023r1
module load cuda/12.1

export APPTAINER_CACHE_DIR="/scratch/chhhorsch/max-proj/explorego/.cache"

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

apptainer exec --nv vizdoom.sif \
     /bin/bash -c "python -m sample_factory.launcher.run\
          --run=sf_examples.vizdoom.experiments.doom_contexts\
          --backend=processes --num_gpus=1 --max_parallel=4\
               --pause_between=0 --experiments_per_gpu=4"
          
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

