"""
Run many experiments with SLURM: hyperparameter sweeps, etc.
This isn't production code, but feel free to use as an example for your SLURM setup.

"""

import os
import time
from os.path import join
from string import Template
from subprocess import PIPE, Popen

from sample_factory.utils.utils import log, str2bool

TIME = '20:00:00'

SBATCH_TEMPLATE_DEFAULT = (
    "#!/bin/bash\n"
    "module load 2023r1\n"
    "module load cuda/12.1\n"
    "export APPTAINER_CACHE_DIR=\"/scratch/chhhorsch/max-proj/explorego/.cache\"\n"
    "previous=$(/usr/bin/nvidia-smi --query-accounted-apps=\'gpu_utilization,mem_utilization,max_memory_usage,time\' --format=\'csv\' | /usr/bin/tail -n \'+2\')\n"
)

def run_slurm_explorego(run_description, args):
    workdir = args.slurm_workdir
    pause_between = args.pause_between

    experiments = run_description.experiments

    log.info("Starting processes with base cmds: %r", [e.cmd for e in experiments])

    if not os.path.exists(workdir):
        log.info("Creating %s...", workdir)
        os.makedirs(workdir)

    # if args.slurm_sbatch_template is not None:
    #     with open(args.slurm_sbatch_template, "r") as template_file:
    #         sbatch_template = template_file.read()
    # else:
    sbatch_template = SBATCH_TEMPLATE_DEFAULT

    log.info("Sbatch template: %s", sbatch_template)

    partition = ""
    if args.slurm_partition is not None:
        partition = f"-p {args.slurm_partition} "

    num_cpus = args.slurm_cpus_per_gpu * args.slurm_gpus_per_job

    experiments = run_description.generate_experiments(args.train_dir)
    sbatch_files = []
    for experiment in experiments:
        cmd, name, *_ = experiment
        cmd = "\napptainer exec --nv vizdoom.sif /bin/bash -c \""+cmd+"\""

        sbatch_fname = f"sbatch_{name}.sh"
        sbatch_fname = join(workdir, sbatch_fname)
        sbatch_fname = os.path.abspath(sbatch_fname)

        file_content = sbatch_template + cmd
        #file_content = Template(sbatch_template).substitute()
        with open(sbatch_fname, "w") as sbatch_f:
            sbatch_f.write(file_content)

        sbatch_files.append(sbatch_fname)

    job_ids = []
    idx = 0
    for sbatch_file in sbatch_files:
        idx += 1
        sbatch_fname = os.path.basename(sbatch_file)
        cmd = f'sbatch --job-name=vizdoom --time={TIME} --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --partition=gpu-a100 --mem-per-cpu=6GB --account=Research-EEMCS-INSY --output={workdir}/{sbatch_fname}-slurm.out --error={workdir}/{sbatch_fname}-slurm.err {sbatch_file}'
        log.info("Executing %s...", cmd)

        if args.slurm_print_only:
            output = idx
        else:
            cmd_tokens = cmd.split()
            process = Popen(cmd_tokens, stdout=PIPE)
            output, err = process.communicate()
            exit_code = process.wait()
            log.info("Output: %s, err: %s, exit code: %r", output, err, exit_code)

            if exit_code != 0:
                log.error("sbatch process failed!")
                time.sleep(5)

        job_id = str(output[20:])
        job_ids.append(job_id)

        time.sleep(pause_between)

    tail_cmd = f"tail -f {workdir}/*.out"
    log.info("Monitor log files using\n\n\t %s \n\n", tail_cmd)

    scancel_cmd = f'scancel {" ".join(job_ids)}'

    log.info("Jobs queued: %r", job_ids)

    log.info("Use this command to cancel your jobs: \n\t %s \n", scancel_cmd)

    with open(join(workdir, "scancel.sh"), "w") as fobj:
        fobj.write(scancel_cmd)

    log.info("Done!")
    return 0
