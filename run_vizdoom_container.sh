apptainer exec --nv vizdoom.sif \
     /bin/bash -c "\
     python -m sample_factory.launcher.run \
          --run=sf_examples.vizdoom.experiments.doom_contexts\
          --backend=processes --num_gpus=1 --max_parallel=4\
               --pause_between=0 --experiments_per_gpu=2"