import argparse
import importlib
import sys

import run_slurm_explorego
from run_slurm_explorego import add_slurm_args, run_slurm


def launcher_argparser(args) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./train_dir", type=str, help="Directory for sub-experiments")
    parser.add_argument(
        "--run",
        default=None,
        type=str,
        help="Name of the python module that describes the run, e.g. sf_examples.vizdoom.experiments.paper_doom_all_basic_envs.py "
        "Run module must be importable in your Python environment. It must define a global variable RUN_DESCRIPTION (see existing run modules for examples).",
    )
    parser.add_argument(
        "--backend",
        default="slurm_explorego",
        choices=["slurm_explorego"],
        help="Launcher backend, use OS multiprocessing by default",
    )
    parser.add_argument("--pause_between", default=1, type=int, help="Pause in seconds between processes")
    parser.add_argument(
        "--experiment_suffix", default="", type=str, help="Append this to the name of the experiment dir"
    )
    parser.add_argument("--num_gpus", default=1)
    parser.add_argument("--max_parallel", default=1)
    parser.add_argument("--experiments_per_gpu", default=1) 


    partial_cfg, _ = parser.parse_known_args(args)
    if partial_cfg.backend == "slurm_explorego":
        parser = add_slurm_args(parser)

    return parser


def parse_args():
    args = launcher_argparser(sys.argv[1:]).parse_args(sys.argv[1:])
    return args


def main():
    launcher_cfg = parse_args()

    try:
        # assuming we're given the full name of the module
        print(launcher_cfg.run)
        run_module = importlib.import_module(f"{launcher_cfg.run}")
    except ImportError as exc:
        print(f"Could not import the run module {exc}")
        return 

    run_description = run_module.RUN_DESCRIPTION
    run_description.experiment_suffix = launcher_cfg.experiment_suffix

    if launcher_cfg.backend == "slurm_explorego":
        run_slurm(run_description, launcher_cfg)


if __name__ == "__main__":
    sys.exit(main())
