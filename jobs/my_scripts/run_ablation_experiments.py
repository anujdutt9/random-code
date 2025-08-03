#!/usr/bin/env python3
"""
Local experiment runner for ablation studies.
This script parses the YAML configuration and runs ablation experiments locally.
"""

import argparse
import yaml
import subprocess
import sys
import os
import glob
from datetime import datetime
from dotenv import load_dotenv

# WandB imports
import wandb

load_dotenv()


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def init_wandb(project_name=None, run_name=None):
    """Initialize WandB run."""
    if run_name is None:
        run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Use environment variables if available
    project = project_name or os.environ.get("WANDB_PROJECT", "cache-steering-experiments")
    entity = os.environ.get("WANDB_ENTITY")
    
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config={
            "timestamp": datetime.now().isoformat(),
        }
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

def run_ablation_experiment(model, task, config, extra_flags, ablate_param, ablate_value, wandb_run=None):
    """Run an ablation experiment with a specific parameter value."""
    model_name = model.split('/')[-1]
    experiment_name = f"{model_name}_{task}_ablation_{ablate_param}_{ablate_value}"
    
    cmd = [
        "python", "eval_steering.py",
        "--model", model,
        "--task", task
    ]
    
    # Create a copy of config and modify the ablation parameter
    ablation_config = config.copy()
    ablation_config[ablate_param] = ablate_value
    
    # Add config parameters
    for key, value in ablation_config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            else:
                cmd.append(f"--no-{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    if extra_flags:
        cmd.extend(extra_flags.split())
    
    # Log experiment start to WandB
    if wandb_run:
        wandb_run.log({
            "experiment_type": "ablation",
            "model": model,
            "task": task,
            "experiment_name": experiment_name,
            "ablate_param": ablate_param,
            "ablate_value": ablate_value,
            "config": ablation_config
        })
    
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

def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments locally")
    parser.add_argument("--config", default="jobs/configs/best_args.yaml", 
                       help="Configuration file")
    parser.add_argument("--task", help="Specific task to run (if not specified, runs all)")
    parser.add_argument("--model", help="Specific model to run (if not specified, runs all)")
    parser.add_argument("--extra-flags", default="--n_runs 1 --eval_type greedy --output_dir my_results/ablation_results --experiment_name ablation", help="Extra flags to pass to evaluation scripts")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    
    # Ablation specific arguments
    parser.add_argument("--ablate-param", required=True, 
                       choices=["n_contrastive_samples", "num_fewshot_examples", "c_values", "c_keys"],
                       help="Parameter to ablate")
    parser.add_argument("--ablate-values", nargs="+", required=True,
                       help="Values to test for the ablation parameter")
    
    # WandB arguments
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", help="WandB project name (uses WANDB_PROJECT env var if not specified)")
    parser.add_argument("--wandb-run-name", help="WandB run name (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # Check environment
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)
    
    # Initialize WandB if enabled
    wandb_run = None
    if args.wandb:
        try:
            wandb_run = init_wandb(args.wandb_project, args.wandb_run_name)
            print(f"WandB logging enabled. Run: {wandb_run.name}")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB: {e}")
            print("Continuing without WandB logging...")
    
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
    
    print(f"Running ablation experiments")
    print(f"Tasks: {tasks}")
    print(f"Models: {models}")
    print(f"Ablation parameter: {args.ablate_param}")
    print(f"Ablation values: {args.ablate_values}")
    print(f"Extra flags: {args.extra_flags}")
    print(f"WandB logging: {args.wandb}")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("my_results/ablation_results", exist_ok=True)
    
    for task in tasks:
        for model in models:
            print(f"\nProcessing {model} on {task}")
            
            # Run ablation experiments for each value
            if task in config and model in config[task]:
                for ablate_value in args.ablate_values:
                    print(f"\n  Running ablation with {args.ablate_param} = {ablate_value}")
                    try:
                        if not args.dry_run:
                            run_ablation_experiment(
                                model, task, config[task][model], 
                                args.extra_flags, args.ablate_param, 
                                ablate_value, wandb_run=wandb_run
                            )
                    except subprocess.CalledProcessError as e:
                        error_msg = f"Error running ablation experiment for {model} on {task} with {args.ablate_param}={ablate_value}: {e}"
                        print(error_msg)
                        if wandb_run:
                            wandb_run.log({"error": error_msg})
            else:
                print(f"No configuration found for {model} on {task}")
    
    print("\n" + "=" * 50)
    print("All ablation experiments completed!")
    
    # Finish WandB run
    if wandb_run:
        wandb_run.finish()
        print("WandB run completed and logged.")

if __name__ == "__main__":
    main() 