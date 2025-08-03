# Instructions for Replicating Experiments

This document provides detailed instructions to replicate all experiments from **"KV Cache Steering for Inducing Reasoning in Small Language Models"**.

---

## Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/MaxBelitsky/cache-steering.git
cd cache-steering
```

2. **Configure Hugging Face credentials**
```bash
export HF_TOKEN=<your_hf_token>
```
Or create a `.env` file using the provided [.env-example](.env).

3. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments
We provide scripts to generate SLURM job files for scalable evaluation and ablations. Below are common usage patterns.

### Baseline (greedy)
```bash
python jobs/scripts/generate_baseline_jobs.py --task arc-oai \
  --extra_flags "--n_runs 1 --encoding_method instruct --eval_type greedy --batch_size 32 --output_dir results/baseline_results" \
  --time 01:00:00
```
Here you need to adjust the `--task` arguments manually to run the experiments on all datasets. The possible values for the datasets can be found under [local_data](/local_data).

### Baseline (sampling)
```bash
python jobs/scripts/generate_baseline_jobs.py --task arc-oai \
  --extra_flags "--n_runs 5 --encoding_method instruct --eval_type sampling --batch_size 32 --output_dir results/baseline_results/sampling" \
  --time 01:00:00
```
Same here – need to adjust the `--task`.

### Baseline + CoT (greedy)
```bash
python jobs/scripts/generate_baseline_jobs.py --task arc-oai \
  --extra_flags "--n_runs 1 --encoding_method instruct --eval_type greedy --batch_size 32 --output_dir results/baseline_results" \
  --time 01:00:00 --include_prefix
```
Same here – need to adjust the `--task`.

### Cache Steering (greedy)
```bash
python jobs/scripts/generate_jobs.py --config jobs/configs/best_args.yaml \
  --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/steering_results --experiment_name steering" \
  --time 01:00:00
```
This script will generate jobs for all model-dataset pairs defined in [jobs/configs/best_args.yaml](jobs/configs/best_args.yaml).

### Cache Steering + CoT (greedy)
```bash
python jobs/scripts/generate_jobs.py --config jobs/configs/best_args.yaml \
  --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/steering_results/prefix --experiment_name steering_prefix" \
  --time 01:00:00 --include_prefix
```

### Cache Steering (sampling)
```bash
python jobs/scripts/generate_jobs.py --config jobs/configs/best_args.yaml \
  --extra_flags "--n_runs 5 --eval_type sampling --output_dir results/steering_results/sampling --experiment_name steering_sampling" \
  --time 02:00:00
```

### Clustering Experiments
```bash
python jobs/scripts/generate_clustering_jobs.py --config jobs/configs/best_args.yaml \
  --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/clustering_results --experiment_name clustering" \
  --time 01:00:00
```

### Ablation Studies
#### Number of Contrastive Samples
```bash
python jobs/scripts/generate_ablation_jobs.py --config jobs/configs/best_args.yaml \
  --ablate_param n_contrastive_samples --ablate_values 100 200 300 400 500 600 700 800 900 1000 \
  --experiment_name n_contrastive_samples_ablation --time 01:00:00 \
  --model "meta-llama/Llama-3.2-1B-Instruct" --task "arc-oai" \
  --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/ablation_results"
```

#### Number of Few-shot Examples
```bash
python jobs/scripts/generate_ablation_jobs.py --config jobs/configs/best_args.yaml \
  --ablate_param num_fewshot_examples --ablate_values 1 3 5 8 10 \
  --experiment_name n_fewshot_ablation --time 01:00:00 \
  --model "meta-llama/Llama-3.2-1B-Instruct" --task "arc-oai" \
  --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/ablation_results"
```

#### c_values
```bash
python jobs/scripts/generate_ablation_jobs.py --config jobs/configs/best_args.yaml \
    --ablate_param c_values --ablate_values 1 2 3 4 5 6 7 8 9 10 \
    --experiment_name c_values_ablation --time 01:00:00 \
    --model "meta-llama/Llama-3.2-1B-Instruct" --task "arc-oai" \
    --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/ablation_results"
```

#### c_keys
```bash
python jobs/scripts/generate_ablation_jobs.py --config jobs/configs/best_args.yaml \
    --ablate_param c_keys --ablate_values 0.0 0.1 0.2 0.3 0.4 \
    --experiment_name c_keys_ablation --time 01:00:00 \
    --model "meta-llama/Llama-3.2-1B-Instruct" --task "arc-oai" \
    --extra_flags "--n_runs 1 --eval_type greedy --output_dir results/ablation_results"
```

### Activation Steering Experiments
To run activation steering experiments use commands from [activation_steering.job](jobs/activation_steering.job).

### Performance Experiments
To run performance experiments use commands from [performance.job](jobs/performance.job).

--- 

## Running Without SLURM
To run any experiment locally, extract the `python ...` command from a generated `.job` file and run it in your configured environment.
