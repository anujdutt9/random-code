#!/usr/bin/env python3
"""
Local experiment runner for cache steering experiments.
This script parses the YAML configuration and runs experiments locally.
Supports multi-GPU model parallelism using Hugging Face Accelerate.
"""

import argparse
import yaml
import subprocess
import sys
import os
import glob
import json
from datetime import datetime
from dotenv import load_dotenv

# WandB imports
import wandb

load_dotenv()


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def parse_extra_flags(extra_flags):
    """Parse extra flags string into a dictionary for better logging."""
    if not extra_flags:
        return {}

    flags_dict = {}
    flags_list = extra_flags.split()
    i = 0
    while i < len(flags_list):
        flag = flags_list[i]
        if flag.startswith('--'):
            flag_name = flag[2:]  # Remove '--'
            # Check if this flag has a value
            if i + 1 < len(flags_list) and not flags_list[i + 1].startswith('--'):
                flags_dict[flag_name] = flags_list[i + 1]
                i += 2
            else:
                # Boolean flag
                flags_dict[flag_name] = True
                i += 1
        else:
            i += 1

    return flags_dict


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


def capture_subprocess_output(cmd, wandb_run=None, env=None):
    """Run subprocess and capture output for WandB logging."""
    print(f"Running: {' '.join(cmd)}")

    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env
    )

    output_lines = []

    # Stream output in real-time
    for line in iter(process.stdout.readline, ''):
        if line:
            line = line.rstrip()
            print(line)  # Print to terminal
            output_lines.append(line)

    # Wait for process to complete
    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    # Log summary to wandb
    if wandb_run:
        wandb_run.log({
            "subprocess_completed": True,
            "total_output_lines": len(output_lines),
            "return_code": return_code
        })

    return output_lines


def upload_output_files(output_dir, wandb_run, experiment_name):
    """Upload output files to WandB as artifacts and log key metrics."""
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist, skipping artifact upload")
        return

    # Find all result files
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))

    if not json_files and not csv_files:
        print(f"No result files found in {output_dir}")
        return

    # Create artifact
    artifact = wandb.Artifact(
        name=f"{experiment_name}_results",
        type="experiment_results",
        description=f"Results for {experiment_name}"
    )

    # Add files to artifact and extract metrics
    uploaded_files = []
    metrics = {}

    for json_file in json_files:
        artifact.add_file(json_file, name=os.path.basename(json_file))
        uploaded_files.append(os.path.basename(json_file))

        # Try to extract metrics from JSON files
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Extract common metrics
                    for key in ['accuracy', 'score', 'final_score', 'avg_score']:
                        if key in data:
                            metrics[f"result_{key}"] = data[key]

                    # Extract any numeric values for logging
                    for key, value in data.items():
                        if isinstance(value, (int, float)) and key not in metrics:
                            metrics[f"result_{key}"] = value
        except Exception as e:
            print(f"Could not extract metrics from {json_file}: {e}")

    for csv_file in csv_files:
        artifact.add_file(csv_file, name=os.path.basename(csv_file))
        uploaded_files.append(os.path.basename(csv_file))

    # Log the artifact
    wandb_run.log_artifact(artifact)

    # Log extracted metrics
    if metrics:
        wandb_run.log(metrics)
        print(f"Logged metrics: {metrics}")

    # Log file info
    wandb_run.log({
        "uploaded_files_count": len(uploaded_files),
        "uploaded_files": uploaded_files
    })

    print(f"Uploaded {len(uploaded_files)} files to WandB as artifact: {artifact.name}")


def run_baseline_experiment(model, task, eval_type, extra_flags, num_fewshot=0, with_prefix=False, enable_wandb=False,
                            wandb_project=None, use_accelerate=False, num_gpus=2, use_fp32=False):
    """Run a baseline experiment."""
    model_name = model.split('/')[-1]
    experiment_name = f"{model_name}_{task}_{eval_type}_baseline"

    if with_prefix:
        experiment_name += "_prefix"

    # Parse extra flags for better logging
    extra_flags_dict = parse_extra_flags(extra_flags)

    # Initialize individual wandb run for this experiment
    wandb_run = None
    if enable_wandb:
        try:
            # Auto-detect device
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

            experiment_config = {
                "experiment_type": "baseline",
                "model": model,
                "model_name": model_name,
                "task": task,
                "eval_type": eval_type,
                "with_prefix": with_prefix,
                "num_fewshot": num_fewshot,
                "device": device,
                "extra_flags_raw": extra_flags,
                "extra_flags": extra_flags_dict,
                "use_accelerate": use_accelerate,
                "num_gpus": num_gpus,
                "use_fp32": use_fp32,
                "parallelism_type": "model_parallelism" if use_accelerate else "single_gpu",
                # Add individual parameters for easier filtering
                "n_runs": extra_flags_dict.get("n_runs", "1"),
                "encoding_method": extra_flags_dict.get("encoding_method", "unknown"),
                "batch_size": extra_flags_dict.get("batch_size", "unknown"),
                "output_dir": extra_flags_dict.get("output_dir", "unknown"),
            }
            wandb_run = init_wandb(wandb_project, experiment_name, experiment_config)
            print(f"WandB run initialized: {wandb_run.name}")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB for {experiment_name}: {e}")

    # Build command using eval_baseline.py's expected arguments ONLY
    if use_accelerate:
        cmd = ["accelerate", "launch"]

        # Add Accelerate configuration for model parallelism
        if num_gpus > 1:
            # Add Accelerate config flags for model parallelism
            cmd.extend([
                "--multi_gpu",
                "--num_processes", str(num_gpus),
                "--main_process_port", "0",  # Use dynamic port assignment
            ])

            # Use FP32 if specified, otherwise use mixed precision
            if use_fp32:
                cmd.extend(["--mixed_precision", "no"])  # No mixed precision = FP32
                print(f"Using FP32 precision with {num_gpus} GPUs for model parallelism")
            else:
                cmd.extend(["--mixed_precision", "bf16"])
                print(f"Using BF16 precision with {num_gpus} GPUs for model parallelism")

            print(f"Letting Accelerate handle GPU assignment automatically")
            print(f"Using dynamic port assignment to avoid conflicts")
            print(f"Working directory: {os.getcwd()}")
        else:
            # Single GPU with Accelerate
            if use_fp32:
                cmd.extend(["--mixed_precision", "no"])
            else:
                cmd.extend(["--mixed_precision", "bf16"])

        # Add the script name
        cmd.append("eval_baseline.py")
    else:
        cmd = ["python", "eval_baseline.py"]

    # Add eval_baseline.py arguments (only the ones it accepts)
    cmd.extend([
        "--experiment-type", "baseline",
        "--task", task,
        "--model", model,
        "--eval-type", eval_type,
    ])

    # Build the complete extra flags string
    all_extra_flags = extra_flags or ""

    # Add num_fewshot if specified
    if num_fewshot != 0:
        all_extra_flags += f" --num_fewshot_prompt {num_fewshot}"

    # Add experiment name
    all_extra_flags += f" --experiment_name {experiment_name}"

    # Add device (auto-detect)
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    all_extra_flags += f" --device {device}"

    # Add the complete extra flags
    if all_extra_flags.strip():
        cmd.extend(["--extra-flags", all_extra_flags.strip()])

    # Add prefix if specified
    if with_prefix:
        cmd.append("--prefix")

    # Add accelerate and GPU configuration
    if use_accelerate:
        cmd.extend(["--use-accelerate", "--num-gpus", str(num_gpus)])
        if use_fp32:
            cmd.append("--use-fp32")

    # Add WandB configuration
    if enable_wandb:
        cmd.append("--wandb")
        if wandb_project:
            cmd.extend(["--wandb-project", wandb_project])
        if experiment_name:
            cmd.extend(["--wandb-run-name", experiment_name])

    env = os.environ.copy()

    try:
        # Run experiment and capture output
        output_lines = capture_subprocess_output(cmd, wandb_run, env=env)

        # Extract output directory from extra_flags
        output_dir = extra_flags_dict.get("output_dir")

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
    parser = argparse.ArgumentParser(description="Run experiments locally")
    parser.add_argument("--config", default="jobs/configs/best_args.yaml",
                        help="Configuration file")
    parser.add_argument("--experiment-type", choices=["baseline"],
                        default="baseline", help="Type of experiment to run")
    parser.add_argument("--task", help="Specific task to run (if not specified, runs all)")
    parser.add_argument("--model", help="Specific model to run (if not specified, runs all)")
    parser.add_argument("--eval-type", choices=["greedy", "sampling"], default="greedy",
                        help="Type of evaluation to run")
    parser.add_argument("--extra-flags",
                        default="--n_runs 1 --encoding_method instruct --batch_size 32 --output_dir my_results/baseline_results",
                        help="Extra flags to pass to evaluation scripts")
    parser.add_argument("--prefix", action="store_true", help="Also run with Chain-of-Thought prefix")

    # Multi-GPU support
    parser.add_argument("--use-accelerate", action="store_true",
                        help="Use Hugging Face Accelerate for multi-GPU support")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="Number of GPUs to use with Accelerate (default: 2)")
    parser.add_argument("--use-fp32", action="store_true",
                        help="Use FP32 precision instead of mixed precision (for fair comparison with original paper)")

    # WandB arguments
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", help="WandB project name (uses WANDB_PROJECT env var if not specified)")
    parser.add_argument("--wandb-run-name", help="WandB run name (auto-generated if not specified)")

    args = parser.parse_args()

    # Check environment
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

    # Validate GPU configuration
    if args.use_accelerate:
        import torch
        available_gpus = torch.cuda.device_count()
        if available_gpus < args.num_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available")
            print(f"Using {available_gpus} GPUs instead")
            args.num_gpus = available_gpus
        elif available_gpus == 0:
            print("Warning: No CUDA GPUs available. Accelerate will use CPU.")
            args.num_gpus = 0

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
    print(f"Use Accelerate: {args.use_accelerate}")
    if args.use_accelerate:
        print(f"Number of GPUs: {args.num_gpus}")
        print(f"Use FP32: {args.use_fp32}")
    print(f"WandB logging: {args.wandb}")
    print("=" * 50)

    # Create output directories
    os.makedirs("my_results/baseline_results", exist_ok=True)

    for task in tasks:
        for model in models:
            print(f"\nProcessing {model} on {task}")

            # Run baseline experiments
            if args.experiment_type in ["baseline", "both"]:
                try:
                    run_baseline_experiment(
                        model, task, args.eval_type, args.extra_flags,
                        with_prefix=args.prefix,
                        enable_wandb=args.wandb,
                        wandb_project=args.wandb_project,
                        use_accelerate=args.use_accelerate,
                        num_gpus=args.num_gpus,
                        use_fp32=args.use_fp32
                    )
                except subprocess.CalledProcessError as e:
                    error_msg = f"Error running baseline experiment for {model} on {task}: {e}"
                    print(error_msg)

    print("\n" + "=" * 50)
    print("All experiments completed!")


if __name__ == "__main__":
    main()