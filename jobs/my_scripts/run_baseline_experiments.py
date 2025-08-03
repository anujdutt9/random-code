#!/usr/bin/env python3
"""
Local experiment runner for cache steering experiments.
This script parses the YAML configuration and runs experiments locally.
"""

import argparse
import yaml
import subprocess
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_baseline_experiment(model, task, extra_flags, num_fewshot=0, with_prefix=False):
    """Run a baseline experiment."""
    model_name = model.split('/')[-1]
    experiment_name = f"{model_name}_{task}_baseline"
    
    # Auto-detect device
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    cmd = [
        "python", "eval_baseline.py",
        "--model", model,
        "--task", task,
        "--num_fewshot_prompt", str(num_fewshot),
        "--experiment_name", experiment_name,
        "--device", device,
    ]
    
    if with_prefix:
        cmd.extend(["--append_prefix_to_prompt"])
        experiment_name += "_prefix"
    
    if extra_flags:
        cmd.extend(extra_flags.split())
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True)

def run_steering_experiment(model, task, config, extra_flags, with_prefix=False):
    """Run a steering experiment."""
    model_name = model.split('/')[-1]
    experiment_name = f"{model_name}_{task}_steering"
    
    cmd = [
        "python", "eval_steering.py",
        "--model", model,
        "--task", task
    ]
    
    # Add config parameters
    for key, value in config.items():
        if key == "prefix" and with_prefix:
            # Handle prefix configuration
            prefix_config = config["prefix"]
            for pkey, pvalue in prefix_config.items():
                if isinstance(pvalue, bool):
                    if pvalue:
                        cmd.append(f"--{pkey}")
                    else:
                        cmd.append(f"--no-{pkey}")
                else:
                    cmd.extend([f"--{pkey}", str(pvalue)])
        elif key != "prefix":  # Skip prefix config for non-prefix runs
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
                else:
                    cmd.append(f"--no-{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    if with_prefix:
        cmd.extend(["--append_prefix_to_prompt", "--add_prefix"])
        experiment_name += "_prefix"

    if extra_flags:
        cmd.extend(extra_flags.split())
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run experiments locally")
    parser.add_argument("--config", default="jobs/configs/best_args.yaml", 
                       help="Configuration file")
    parser.add_argument("--experiment-type", choices=["baseline", "steering", "both"], 
                       default="baseline", help="Type of experiment to run")
    parser.add_argument("--task", help="Specific task to run (if not specified, runs all)")
    parser.add_argument("--model", help="Specific model to run (if not specified, runs all)")
    parser.add_argument("--extra-flags", default="--n_runs 1 --encoding_method instruct --eval_type greedy --batch_size 32 --output_dir my_results/baseline_results", help="Extra flags to pass to evaluation scripts")
    parser.add_argument("--prefix", action="store_true", help="Also run with Chain-of-Thought prefix")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    
    args = parser.parse_args()
    
    # Check environment
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)
    
    # Load configuration
    if args.experiment_type in ["steering", "both"]:
        config = load_config(args.config)
    
    # Define available models and tasks
    all_models = [
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct", 
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "Qwen/Qwen2-0.5B-Instruct"
    ]
    
    all_tasks = ["arc-oai", "csqa-oai", "gsm8k-oai", "piqa-oai"]
    
    # Filter models and tasks
    models = [args.model] if args.model else all_models
    tasks = [args.task] if args.task else all_tasks
    
    print(f"Running {args.experiment_type} experiments")
    print(f"Tasks: {tasks}")
    print(f"Models: {models}")
    print(f"Extra flags: {args.extra_flags}")
    print(f"With prefix: {args.prefix}")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("my_results/baseline_results", exist_ok=True)
    os.makedirs("my_results/steering_results", exist_ok=True)
    
    for task in tasks:
        for model in models:
            print(f"\nProcessing {model} on {task}")
            
            # Run baseline experiments
            if args.experiment_type in ["baseline", "both"]:
                try:
                    if not args.dry_run:
                        run_baseline_experiment(model, task, args.extra_flags)
                    if args.prefix and not args.dry_run:
                        run_baseline_experiment(model, task, args.extra_flags, with_prefix=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running baseline experiment for {model} on {task}: {e}")
            
            # Run steering experiments
            if args.experiment_type in ["steering", "both"]:
                if task in config and model in config[task]:
                    try:
                        if not args.dry_run:
                            run_steering_experiment(model, task, config[task][model], args.extra_flags)
                        if args.prefix and not args.dry_run:
                            run_steering_experiment(model, task, config[task][model], args.extra_flags, with_prefix=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running steering experiment for {model} on {task}: {e}")
                else:
                    print(f"No configuration found for {model} on {task}")
    
    print("\n" + "=" * 50)
    print("All experiments completed!")

if __name__ == "__main__":
    main()


# python eval_baseline.py --model HuggingFaceTB/SmolLM2-360M-Instruct --task arc-oai
# --num_fewshot_prompt 0 --experiment_name SmolLM2-360M-Instruct_arc-oai_baseline
# --device cuda --n_runs 1 --encoding_method instruct --eval_type greedy
# --batch_size 32 --output_dir my_results/baseline_results