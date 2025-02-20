"""
Run many experiments with SLURM: hyperparameter sweeps, etc.
This isn't production code, but feel free to use as an example for your SLURM setup.

"""

import os
import time
from os.path import join
from subprocess import PIPE, Popen


TIME = '20:00:00'
CPUS_PER_TASK = '16'
PARTITION = 'insy,general'
ACCOUNT = 'ewi-insy-sdm'

SBATCH_TEMPLATE_DEFAULT = (
    "#!/bin/bash\n"
    "module load 2023r1\n"
    "module load cuda/12.1\n"
    "export APPTAINER_CACHE_DIR=\"/scratch/chhhorsch/max-proj/explorego/.cache\"\n"
    "previous=$(/usr/bin/nvidia-smi --query-accounted-apps=\'gpu_utilization,mem_utilization,max_memory_usage,time\' --format=\'csv\' | /usr/bin/tail -n \'+2\')\n"
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true",):
        return True
    elif isinstance(v, str) and v.lower() in ("false",):
        return False
    else:
        return False
    
def add_slurm_args(parser):
    parser.add_argument("--slurm_gpus_per_job", default=1, type=int, help="GPUs in a single SLURM process")
    parser.add_argument(
        "--slurm_cpus_per_gpu", default=14, type=int, help="Max allowed number of CPU cores per allocated GPU"
    )
    parser.add_argument(
        "--slurm_print_only", default=False, type=str2bool, help="Just print commands to the console without executing"
    )
    parser.add_argument(
        "--slurm_workdir",
        default=None,
        type=str,
        help="Optional workdir. Used by slurm launcher to store logfiles etc.",
    )
    parser.add_argument(
        "--slurm_partition",
        default=None,
        type=str,
        help='Adds slurm partition, i.e. for "gpu" it will add "-p gpu" to sbatch command line',
    )

    parser.add_argument(
        "--slurm_sbatch_template",
        default=None,
        type=str,
        help="Commands to run before the actual experiment (i.e. activate conda env, etc.)",
    )

    parser.add_argument(
        "--slurm_timeout",
        default="0",
        type=str,
        help="Time to run jobs before timing out job and requeuing the job. Defaults to 0, which does not time out the job",
    )

    return parser

def run_slurm(run_description, args):
    workdir = args.slurm_workdir
    pause_between = args.pause_between

    experiments = run_description.experiments

    print("Starting processes with base cmds: %r", [e.cmd for e in experiments])

    if not os.path.exists(workdir):
        print("Creating %s...", workdir)
        os.makedirs(workdir)

    sbatch_template = SBATCH_TEMPLATE_DEFAULT

    print("Sbatch template: %s", sbatch_template)

    experiments = run_description.generate_experiments(args.train_dir)
    sbatch_files = []
    for experiment in experiments:
        cmd, name, *_ = experiment
        cmd = "\napptainer exec --nv vizdoom.sif /bin/bash -c \""+cmd+"\""

        sbatch_fname = f"sbatch_{name}.sh"
        sbatch_fname = join(workdir, sbatch_fname)
        sbatch_fname = os.path.abspath(sbatch_fname)

        file_content = sbatch_template + cmd
        with open(sbatch_fname, "w") as sbatch_f:
            sbatch_f.write(file_content)

        sbatch_files.append(sbatch_fname)

    job_ids = []
    idx = 0
    for sbatch_file in sbatch_files:
        idx += 1
        sbatch_fname = os.path.basename(sbatch_file)
        cmd = f'sbatch --job-name=vizdoom --time={TIME} --ntasks=1 --gpus-per-task=1 --cpus-per-task={CPUS_PER_TASK} --partition={PARTITION} --mem-per-cpu=6GB --account={ACCOUNT} --output={workdir}/{sbatch_fname}-slurm.out --error={workdir}/{sbatch_fname}-slurm.err {sbatch_file}'
        print("Executing %s...", cmd)

        if args.slurm_print_only:
            output = idx
        else:
            cmd_tokens = cmd.split()
            process = Popen(cmd_tokens, stdout=PIPE)
            output, err = process.communicate()
            exit_code = process.wait()
            print("Output: %s, err: %s, exit code: %r", output, err, exit_code)

            if exit_code != 0:
                print("sbatch process failed!")
                time.sleep(5)

        #job_id = str(output[20:])
        job_id = str(output)
        job_ids.append(job_id)

        time.sleep(pause_between)

    tail_cmd = f"tail -f {workdir}/*.out"
    print("Monitor log files using\n\n\t %s \n\n", tail_cmd)

    scancel_cmd = f'scancel {" ".join(job_ids)}'

    print("Jobs queued: %r", job_ids)

    print("Use this command to cancel your jobs: \n\t %s \n", scancel_cmd)

    with open(join(workdir, "scancel.sh"), "w") as fobj:
        fobj.write(scancel_cmd)

    print("Done!")
    return 0
