"""
Generate SLURM job scripts from configuration files.

Example usage:
- [Clustering]: python jobs/scripts/generate_clustering_jobs.py --config jobs/configs/best_args.yaml --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/clustering_results --experiment_name greedy_clustering" --time 01:00:00
"""

from copy import copy
import yaml
import os
import argparse

INDENT = "\t"*6
# Load YAML configuration
TEMPLATE_PATH = "jobs/templates/steering.job"
OUTPUT_DIR = "jobs/generated/clustering/"

DEFAULT_CLUSTERING_LAYERS = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "cluster_layer_id": 8,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "cluster_layer_id": 14,
    },
    "HuggingFaceTB/SmolLM-360M-Instruct": {
        "cluster_layer_id": 16,
    },
}


def add_argument(key, value):
    if isinstance(value, bool):
        return f" \\\n{INDENT}--{key}" if value else ""
    return f" \\\n{INDENT}--{key} {value}"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate SLURM job scripts from config")
    parser.add_argument("--config", type=str, default="jobs/configs/best_args.yaml", help="Path to the configuration file")
    parser.add_argument("--extra_flags", type=str, default="", help="Additional flags to pass to the job")
    parser.add_argument("--time", type=str, default="04:00:00", help="Time limit for the job")
    parser.add_argument("--experiment_name", type=str, default="best_experiment", help="Experiment name")
    
    # Clustering specific arguments
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--n_contrastive_samples", type=int, default=1000, help="Number of contrastive samples")
    parser.add_argument("--cluster_on", type=str, default="activations", help="Cluster on which types of vectors")
    parser.add_argument("--cluster_layer_id", type=int, required=False, help="Layer ID for clustering")

    args = parser.parse_args()

    extra_flags = args.extra_flags.strip()  # Ensure proper formatting

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Read template and fill values
    with open(TEMPLATE_PATH, "r") as template_file:
        template = template_file.read()

    # Iterate over all configs
    for task, task_params in config.items():
        for model, model_params in task_params.items():
            
            base_template = copy(template)
            base_dict, prefix_dict = {}, {}

            for key, value in model_params.items():
                if isinstance(value, dict):
                    prefix_dict = value
                else:
                    base_dict[key] = value

            # Add clustering specific arguments
            if args.cluster_layer_id is None:
                cluster_layer_id = DEFAULT_CLUSTERING_LAYERS.get(model, {}).get("cluster_layer_id")
            else:
                cluster_layer_id = args.cluster_layer_id

            base_dict["n_clusters"] = args.n_clusters
            base_dict["n_contrastive_samples"] = args.n_contrastive_samples
            base_dict["cluster_on"] = args.cluster_on
            base_dict["cluster_layer_id"] = cluster_layer_id
            base_dict["aggregation_method"] = "clustering"

            # Add base arguments
            for key, value in base_dict.items():
                base_template += add_argument(key, value)
            base_template += (f" \\\n{INDENT}" + extra_flags) if extra_flags else ""

            # Job file naming
            model_name = model.split("/")[-1]
            output_dir = os.path.join(OUTPUT_DIR, task)
            os.makedirs(output_dir, exist_ok=True)
            # Create job file
            job_file = os.path.join(output_dir, f"{model_name}.job")

            job_script = base_template.format(
                TIME=args.time,
                EXPERIMENT_NAME=args.experiment_name,
                MODEL_NAME=model_name,
                MODEL=model,
                TASK=task,
                JOB_NAME=f"{task[:2].upper()}_{model_name[:1].upper()}",
            )
            print(f"Generated job: {job_file}")

            # Write to job file
            with open(job_file, "w") as output_file:
                output_file.write(job_script)

            # Add prefix arguments
            if prefix_dict:
                prefix_template = copy(template)
                base_dict.update(prefix_dict)
                for key, value in base_dict.items():
                    prefix_template += add_argument(key, value)
                prefix_template += (f" \\\n{INDENT}" + extra_flags) if extra_flags else ""

                prefix_output_dir = os.path.join(OUTPUT_DIR, task, "prefix")
                os.makedirs(prefix_output_dir, exist_ok=True)
                prefix_job_file = os.path.join(prefix_output_dir, f"{model_name}.job")
                prefix_job_script = prefix_template.format(
                    TIME=args.time,
                    EXPERIMENT_NAME=args.experiment_name + "_prefix",
                    MODEL_NAME=model_name,
                    MODEL=model,
                    TASK=task,
                    JOB_NAME=f"{task[:2].upper()}_{model_name[:1].upper()}",
                )

                # Write to job file
                # with open(prefix_job_file, "w") as output_file:
                #     output_file.write(prefix_job_script)

                # print(f"Generated job: {prefix_job_file}")
