"""
Python script to generate SLURM job scripts for running baseline experiments.

Example usage:
- [Baseline]: python jobs/scripts/generate_baseline_jobs.py --task arc-oai --extra_flags "--n_runs 1 --encoding_method instruct --eval_type greedy --batch_size 32 --output_dir results/baseline_results" --time 01:00:00
- [Sampling baseline]: python jobs/scripts/generate_baseline_jobs.py --task arc-oai --extra_flags "--n_runs 5 --encoding_method instruct --eval_type sampling --batch_size 32 --output_dir results/baseline_results/sampling" --time 01:00:00
"""

from copy import copy
import os
import argparse
from itertools import product

INDENT = "\t"*6
# Load YAML configuration
TEMPLATE_PATH = "jobs/templates/baseline.job"
OUTPUT_DIR = "jobs/generated/baseline/"

NUM_FEWSHOTS = [0]
MODELS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2-0.5B-Instruct",
]
COMBINATIONS = product(NUM_FEWSHOTS, MODELS)


def add_argument(key, value):
    if isinstance(value, bool):
        return f" \\\n{INDENT}--{key}" if value else ""
    if isinstance(value, str):
        value = f'"{value}"'
    return f" \\\n{INDENT}--{key} {value}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SLURM job scripts ")
    parser.add_argument("--extra_flags", type=str, default="", help="Additional flags to pass to the job")
    parser.add_argument("--time", type=str, default="04:00:00", help="Time limit for the job")
    parser.add_argument("--task", type=str, help="Task to generate jobs for")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR + f"{args.task}/", exist_ok=True)
    extra_flags = args.extra_flags.strip()

    # Read template and fill values
    with open(TEMPLATE_PATH, "r") as template_file:
        template = template_file.read()

    # Iterate over all configs
    for num_fewshot, model in COMBINATIONS:

        # Job file naming
        model_name = model.split("/")[-1]
        job_file = os.path.join(OUTPUT_DIR + f"{args.task}/", f"{model_name}_{num_fewshot}_baseline.job")
        prefix_job_file = os.path.join(OUTPUT_DIR + f"{args.task}/", f"{model_name}_{num_fewshot}_prefix_baseline.job")
        experiment_name = f"{model_name}_{args.task}_baseline"

        current_template = copy(template)

        job_script = current_template.format(
            TIME=args.time,
            EXPERIMENT_NAME=experiment_name,
            MODEL_NAME=model_name,
            MODEL=model,
            TASK=args.task,
            NUM_FEWSHOT=num_fewshot,
        )
        job_script += (f" \\\n{INDENT}" + extra_flags) if extra_flags else ""

        # Add prefix arguments
        prefix_job_script = copy(job_script)
        prefix_job_script += add_argument("append_prefix_to_prompt", True)

        # Write to job file
        with open(job_file, "w") as output_file:
            output_file.write(job_script)

        with open(prefix_job_file, "w") as output_file:
            output_file.write(prefix_job_script)

        print(f"Generated job: {job_file}")
        print(f"Generated job: {prefix_job_file}")
