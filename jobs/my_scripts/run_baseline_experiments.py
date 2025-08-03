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
import glob
import json
import tempfile
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


def create_deepspeed_config(num_gpus=2, fp16=False, zero_stage=1):
    """Create a temporary DeepSpeed configuration file."""
    config = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 100
            }
        },
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 2000,
        "wall_clock_breakdown": False,
        "fp16": {
            "enabled": fp16,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 32,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": False
        }
    }
    
    # Create temporary file
    fd, config_path = tempfile.mkstemp(suffix='.json', prefix='deepspeed_config_')
    os.close(fd)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def detect_gpu_memory():
    """Detect available GPU memory and number of GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_memory = []
            for i in range(num_gpus):
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                gpu_memory.append(memory)
            return num_gpus, gpu_memory
        else:
            return 0, []
    except ImportError:
        return 0, []


def should_use_deepspeed(model_name, num_gpus, gpu_memory):
    """Determine if DeepSpeed should be used based on model size and GPU memory."""
    # Model size estimates in GB (approximate)
    model_sizes = {
        "SmolLM2-360M-Instruct": 0.7,
        "Llama-3.2-1B-Instruct": 2.0,
        "Llama-3.2-3B-Instruct": 6.0,
        "Llama-3.1-8B-Instruct": 16.0,
        "Phi-4-mini-instruct": 4.0,
        "Qwen2-0.5B-Instruct": 1.0
    }
    
    # Extract model name from full path
    model_short_name = model_name.split('/')[-1]
    estimated_size = model_sizes.get(model_short_name, 8.0)  # Default to 8GB if unknown
    
    # Check if model fits on single GPU
    if num_gpus >= 2 and gpu_memory:
        single_gpu_memory = gpu_memory[0]
        # Use DeepSpeed if model size > 80% of single GPU memory
        if estimated_size > single_gpu_memory * 0.8:
            return True
    
    return False


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
                            wandb_project=None, use_deepspeed=False, use_fp16=False, num_gpus=2):
    """Run a baseline experiment."""
    model_name = model.split('/')[-1]
    experiment_name = f"{model_name}_{task}_{eval_type}_baseline"

    if with_prefix:
        experiment_name += "_prefix"

    # Parse extra flags for better logging
    extra_flags_dict = parse_extra_flags(extra_flags)

    # Detect GPU setup
    num_available_gpus, gpu_memory = detect_gpu_memory()
    
    # Determine if we should use DeepSpeed
    if use_deepspeed or should_use_deepspeed(model, num_available_gpus, gpu_memory):
        use_deepspeed = True
        print(f"Using DeepSpeed for {model} (estimated size may exceed single GPU memory)")
    
    # Initialize individual wandb run for this experiment
    wandb_run = None
    deepspeed_config_path = None
    
    if enable_wandb:
        try:
            # Build command first to include in config
            cmd = build_experiment_command(model, task, eval_type, experiment_name, num_fewshot, 
                                         with_prefix, extra_flags, use_deepspeed, use_fp16, num_gpus)

            experiment_config = {
                "experiment_type": "baseline",
                "model": model,
                "model_name": model_name,
                "task": task,
                "eval_type": eval_type,
                "with_prefix": with_prefix,
                "num_fewshot": num_fewshot,
                "use_deepspeed": use_deepspeed,
                "use_fp16": use_fp16,
                "num_gpus": num_gpus,
                "num_available_gpus": num_available_gpus,
                "gpu_memory": gpu_memory,
                "command": " ".join(cmd),
                "cmd_length": len(cmd),
                "extra_flags_raw": extra_flags,
                "extra_flags": extra_flags_dict,
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
    else:
        # Build command
        cmd = build_experiment_command(model, task, eval_type, experiment_name, num_fewshot, 
                                     with_prefix, extra_flags, use_deepspeed, use_fp16, num_gpus)

    try:
        # Create DeepSpeed config if needed
        if use_deepspeed:
            deepspeed_config_path = create_deepspeed_config(num_gpus, use_fp16)
            print(f"Created DeepSpeed config: {deepspeed_config_path}")

        # Run experiment and capture output
        output_lines = capture_subprocess_output(cmd, wandb_run)

        # Extract output directory from extra_flags
        output_dir = extra_flags_dict.get("output_dir")

        # Upload results to WandB
        if wandb_run and output_dir:
            upload_output_files(output_dir, wandb_run, experiment_name)

        return output_lines
    finally:
        # Clean up DeepSpeed config file
        if deepspeed_config_path and os.path.exists(deepspeed_config_path):
            os.unlink(deepspeed_config_path)
        
        # Always finish the wandb run
        if wandb_run:
            wandb_run.finish()
            print(f"WandB run completed: {experiment_name}")


def build_experiment_command(model, task, eval_type, experiment_name, num_fewshot, with_prefix, extra_flags, 
                           use_deepspeed, use_fp16, num_gpus):
    """Build the command for running the experiment."""
    if use_deepspeed:
        # Use DeepSpeed for multi-GPU inference
        cmd = [
            "deepspeed", "--num_gpus", str(num_gpus), "eval_baseline.py",
            "--model", model,
            "--task", task,
            "--num_fewshot_prompt", str(num_fewshot),
            "--experiment_name", experiment_name,
            "--eval_type", eval_type,
        ]
        
        if use_fp16:
            cmd.extend(["--fp16"])
    else:
        # Use regular Python with single GPU
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        cmd = [
            "python", "eval_baseline.py",
            "--model", model,
            "--task", task,
            "--num_fewshot_prompt", str(num_fewshot),
            "--experiment_name", experiment_name,
            "--device", device,
            "--eval_type", eval_type,
        ]

    if with_prefix:
        cmd.extend(["--append_prefix_to_prompt"])

    if extra_flags:
        cmd.extend(extra_flags.split())
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run experiments locally")
    parser.add_argument("--config", default="jobs/configs/best_args.yaml",
                        help="Configuration file")
    parser.add_argument("--experiment-type", choices=["baseline"],
                        default="baseline", help="Type of experiment to run")
    parser.add_argument("--task", help="Specific task to run (if not specified, runs all)")
    parser.add_argument("--model", help="Specific model to run (if not specified, runs all)")
    parser.add_argument("--eval-type", choices=["greedy", "sampling"], default="greedy", help="Type of evaluation to run")
    parser.add_argument("--extra-flags",
                        default="--n_runs 1 --encoding_method instruct --batch_size 32 --output_dir my_results/baseline_results",
                        help="Extra flags to pass to evaluation scripts")
    parser.add_argument("--prefix", action="store_true", help="Also run with Chain-of-Thought prefix")

    # DeepSpeed arguments
    parser.add_argument("--use-deepspeed", action="store_true", 
                        help="Force use of DeepSpeed for multi-GPU inference")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use FP16 precision (default is FP32)")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="Number of GPUs to use with DeepSpeed")

    # WandB arguments
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", help="WandB project name (uses WANDB_PROJECT env var if not specified)")
    parser.add_argument("--wandb-run-name", help="WandB run name (auto-generated if not specified)")

    args = parser.parse_args()

    # Check environment
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)

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
    print(f"Use DeepSpeed: {args.use_deepspeed}")
    print(f"Use FP16: {args.fp16}")
    print(f"Number of GPUs: {args.num_gpus}")
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
                    run_baseline_experiment(model, task, args.eval_type, args.extra_flags, with_prefix=args.prefix,
                                                enable_wandb=args.wandb, wandb_project=args.wandb_project,
                                                use_deepspeed=args.use_deepspeed, use_fp16=args.fp16, 
                                                num_gpus=args.num_gpus)
                except subprocess.CalledProcessError as e:
                    error_msg = f"Error running baseline experiment for {model} on {task}: {e}"
                    print(error_msg)

    print("\n" + "=" * 50)
    print("All experiments completed!")


if __name__ == "__main__":
    main()