#!/usr/bin/env python3
"""
Local experiment runner for cache steering experiments.
This script parses the YAML configuration and runs steering experiments locally.
"""

import argparse
import yaml
import subprocess
import sys
import os
import glob
import torch
from datetime import datetime
from dotenv import load_dotenv

# WandB imports
import wandb

load_dotenv()


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def init_wandb(project_name=None, run_name=None, experiment_config=None):
    """Initialize WandB run."""
    if run_name is None:
        run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Use environment variables if available
    project = project_name or os.environ.get("WANDB_PROJECT", "cache-steering-experiments")
    entity = os.environ.get("WANDB_ENTITY")
    
    config = {
        "timestamp": datetime.now().isoformat(),
    }
    if experiment_config:
        config.update(experiment_config)
    
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config
    )
    return wandb.run

def capture_subprocess_output(cmd, wandb_run=None):
    """Run subprocess and capture output for WandB logging."""
    print(f"Running: {' '.join(cmd)}")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    output_lines = []
    
    # Stream output in real-time
    for line in iter(process.stdout.readline, ''):
        if line:
            line = line.rstrip()
            print(line)  # Print to terminal
            output_lines.append(line)
            
            # Log to WandB if available
            if wandb_run:
                wandb_run.log({"terminal_output": line})
    
    # Wait for process to complete
    return_code = process.wait()
    
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    
    return output_lines

def upload_output_files(output_dir, wandb_run, experiment_name):
    """Upload output files to WandB as artifacts."""
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist, skipping artifact upload")
        return
    
    # Find all JSON files in the output directory
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {output_dir}")
        return
    
    # Create artifact
    artifact = wandb.Artifact(
        name=f"{experiment_name}_results",
        type="experiment_results",
        description=f"Results for {experiment_name}"
    )
    
    # Add files to artifact
    for json_file in json_files:
        artifact.add_file(json_file, name=os.path.basename(json_file))
    
    # Log the artifact
    wandb_run.log_artifact(artifact)
    print(f"Uploaded {len(json_files)} files to WandB as artifact: {artifact.name}")

def run_steering_experiment(model, task, eval_type, config, extra_flags, with_prefix=False, enable_wandb=False, wandb_project=None):
    """Run a steering experiment."""
    model_name = model.split('/')[-1]
    experiment_name = f"{model_name}_{task}_{eval_type}_steering"

    # Auto-detect device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    cmd = [
        "python", "eval_steering.py",
        "--model", model,
        "--task", task,
        "--device", device,
        "--eval_type", eval_type,
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

    # Initialize WandB for this specific experiment
    wandb_run = None
    if enable_wandb:
        try:
            # Auto-detect device
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

            experiment_config = {
                "experiment_type": "steering",
                "model": model,
                "model_name": model_name,
                "task": task,
                "eval_type": eval_type,
                "with_prefix": with_prefix,
                "device": device,
                "command": " ".join(cmd),
                "config": config
            }
            wandb_run = init_wandb(wandb_project, experiment_name, experiment_config)
            print(f"WandB run initialized: {wandb_run.name}")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB for {experiment_name}: {e}")
    
    try:
        # Run experiment and capture output
        output_lines = capture_subprocess_output(cmd, wandb_run)
        
        # Extract output directory from extra_flags
        output_dir = None
        if extra_flags:
            flags_list = extra_flags.split()
            for i, flag in enumerate(flags_list):
                if flag == "--output_dir" and i + 1 < len(flags_list):
                    output_dir = flags_list[i + 1]
                    break
        
        # Upload results to WandB
        if wandb_run and output_dir:
            upload_output_files(output_dir, wandb_run, experiment_name)
        
        return output_lines
    finally:
        # Always finish the wandb run
        if wandb_run:
            wandb_run.finish()
            print(f"WandB run completed: {experiment_name}")

def main():
    parser = argparse.ArgumentParser(description="Run steering experiments locally")
    parser.add_argument("--config", default="jobs/configs/best_args.yaml", 
                       help="Configuration file")
    parser.add_argument("--task", help="Specific task to run (if not specified, runs all)")
    parser.add_argument("--model", help="Specific model to run (if not specified, runs all)")
    parser.add_argument("--eval-type", choices=["greedy", "sampling"], default="greedy", help="Type of evaluation to run")
    parser.add_argument("--extra-flags", default="--n_runs 1 --output_dir my_results/steering_results --experiment_name steering", help="Extra flags to pass to evaluation scripts")
    parser.add_argument("--prefix", action="store_true", help="Also run with Chain-of-Thought prefix")

    # WandB arguments
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", help="WandB project name (uses WANDB_PROJECT env var if not specified)")
    parser.add_argument("--wandb-run-name", help="WandB run name (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # Check environment
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)
    

    
    # Load configuration
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
    
    print(f"Running steering experiments")
    print(f"Tasks: {tasks}")
    print(f"Models: {models}")
    print(f"Extra flags: {args.extra_flags}")
    print(f"With prefix: {args.prefix}")
    print(f"WandB logging: {args.wandb}")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("my_results/steering_results", exist_ok=True)
    
    for task in tasks:
        for model in models:
            print(f"\nProcessing {model} on {task}")
            
            # Run steering experiments
            if task in config and model in config[task]:
                try:
                    run_steering_experiment(model, task, args.eval_type, config[task][model], args.extra_flags, with_prefix=args.prefix, enable_wandb=args.wandb, wandb_project=args.wandb_project)
                except subprocess.CalledProcessError as e:
                    error_msg = f"Error running steering experiment for {model} on {task}: {e}"
                    print(error_msg)
            else:
                print(f"No configuration found for {model} on {task}")
    
    print("\n" + "=" * 50)
    print("All steering experiments completed!")

if __name__ == "__main__":
    main()
