"""
Python script to generate SLURM job scripts for running ablation experiments.

Example usage:
python jobs/scripts/generate_ablation_jobs.py \
    --config jobs/configs/best_args.yaml \
    --ablate_param n_contrastive_samples \
    --ablate_values 100 200 300 400 500 \
    --experiment_name temp_ablation \
    --time 02:00:00 \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --task "arc-oai" \
    --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/ablation_results --experiment_name ablation"
"""

import os
import argparse
import yaml
from copy import copy

INDENT = "\t" * 7
TEMPLATE_PATH = "jobs/templates/ablation.job"
OUTPUT_DIR = "jobs/generated/ablations/"


def add_argument(key, value):
    """Format a CLI argument for SLURM script."""
    if isinstance(value, bool):
        return f" \\\n{INDENT}--{key}" if value else ""
    if isinstance(value, str):
        value = f'"{value}"'
    return f" \\\n{INDENT}--{key} {value}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLURM job arrays for ablation studies.")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config file")
    parser.add_argument("--ablate_param", type=str, required=True, help="Parameter name to vary")
    parser.add_argument("--ablate_values", nargs="+", required=True, help="List of values to ablate")
    parser.add_argument("--time", type=str, default="02:00:00", help="Time for SLURM jobs")
    parser.add_argument("--experiment_name", type=str, default="ablation_study", help="Experiment name")
    parser.add_argument("--extra_flags", type=str, default="", help="Extra flags to append")
    parser.add_argument("--model", type=str, help="Restrict to a specific model")
    parser.add_argument("--task", type=str, help="Restrict to a specific task")
    args = parser.parse_args()

    # Load base config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Load SLURM template
    with open(TEMPLATE_PATH, "r") as template_file:
        template = template_file.read()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    value_file_path = os.path.join(OUTPUT_DIR, f"{args.ablate_param}_values.txt")

    # Write ablation values to file
    with open(value_file_path, "w") as f:
        for val in args.ablate_values:
            f.write(str(val) + "\n")

    for task, task_params in config.items():
        if args.task and task != args.task:
            continue

        for model, model_params in task_params.items():
            if args.model and model != args.model:
                continue

            model_name = model.split("/")[-1]
            task_dir = os.path.join(OUTPUT_DIR, task)
            os.makedirs(task_dir, exist_ok=True)

            job_file = os.path.join(task_dir, f"{model_name}_{args.ablate_param}.job")

            # Prepare static arguments from config (excluding ablate_param)
            cli_args = ""
            for key, value in model_params.items():
                if key == args.ablate_param:
                    continue
                if key == "prefix":
                    continue
                cli_args += add_argument(key, value)

            # Add extra_flags
            if args.extra_flags:
                cli_args += f" \\\n{INDENT}{args.extra_flags.strip()}"

            # Fill template
            job_script = copy(template).format(
                TIME=args.time,
                EXPERIMENT_NAME=args.experiment_name,
                MODEL_NAME=model_name,
                MODEL=model,
                TASK=task,
                JOB_NAME=f"ABL_{args.ablate_param.upper()}_{model_name[:1].upper()}",
                N=len(args.ablate_values) - 1,
                PARAM=args.ablate_param,
                EXTRA_FLAGS=cli_args
            )

            with open(job_file, "w") as f:
                f.write(job_script)

            print(f"Generated: sbatch {job_file}")
