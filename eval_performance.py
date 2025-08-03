import os
import time
from argparse import ArgumentParser, BooleanOptionalAction
from copy import copy
import json
import random

import torch
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

from src.data.contrastive_dataset import ContrastiveDatasetConstructor
from src.data.evaluation_dataset import EvaluationDatasetConstructor
from src.data.loader import load_task_dataset
from src.steering.config import SteeringConfig
from src.utils.constants import LLAMA_CHAT_TEMPLATE
from src.utils.logging_setup import log_stream, logger
from src.steering.cache_steering import extract_steering_kv, generate_with_cache_steering


def save_results(results_path, all_results):
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)


def generate_baseline(model, tokenizer, dataset, device, batch_size, generation_kwargs):

    n_generated_tokens = []
    generation_times = []
    for batch in tqdm(dataset.iter(batch_size=batch_size), desc="Generating baseline"):
        inputs = tokenizer(
            batch["input"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs, use_cache=True)
        end_time = time.time()

        generation_time = end_time - start_time
        generated_tokens = (outputs.shape[-1] - inputs['input_ids'].shape[-1]) * batch_size
        n_generated_tokens.append(generated_tokens)
        generation_times.append(generation_time)

    return {
        "n_generated_tokens": n_generated_tokens,
        "generation_times": generation_times,
    }


def generate_cache_steering(model, tokenizer, dataset, device, batch_size, generation_kwargs, steering_kv, steering_config):
    n_generated_tokens = []
    generation_times = []
    for batch in tqdm(dataset.iter(batch_size=batch_size), desc="Generating with cache steering"):
        inputs = tokenizer(
            batch["input"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = generate_with_cache_steering(
                model,
                inputs["input_ids"],
                steering_kv,
                steering_config=steering_config,
                attention_mask=inputs["attention_mask"],
                use_cache=True,
                **generation_kwargs,
            )
        end_time = time.time()
        generation_time = end_time - start_time

        generated_tokens = (outputs.shape[-1] - inputs['input_ids'].shape[-1]) * batch_size
        n_generated_tokens.append(generated_tokens)
        generation_times.append(generation_time)

    return {
        "n_generated_tokens": n_generated_tokens,
        "generation_times": generation_times,
    }


def generate_activation_steering(
    model,
    tokenizer,
    dataset,
    device,
    batch_size,
    generation_kwargs,
    steering_vector,
    multiplier,
    continuous=True,
):
    n_generated_tokens = []
    generation_times = []

    for batch in tqdm(dataset.iter(batch_size=batch_size), desc="Generating with activation steering"):
        inputs = tokenizer(
            batch["input"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        ).to(device)

        min_token_index = -1 if continuous else inputs["input_ids"].shape[1] - 1

        # Apply the steering vector to the model
        handle = steering_vector.patch_activations(
            model=model,
            multiplier=multiplier,
            min_token_index=min_token_index,  # Apply to the last token only
        )

        try:
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                    use_cache=False if continuous else True,
                )
            end_time = time.time()
            generation_time = end_time - start_time
        finally:
            # Remove the patch from the model
            handle.remove()

        generated_tokens = (outputs.shape[-1] - inputs['input_ids'].shape[-1]) * batch_size
        n_generated_tokens.append(generated_tokens)
        generation_times.append(generation_time)

    # Remove the patch from the model
    handle.remove()

    return {
        "n_generated_tokens": n_generated_tokens,
        "generation_times": generation_times,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="performance_results")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rerun_existing", action=BooleanOptionalAction, default=False)
    parser.add_argument("--n_runs", default=3, type=int)

    args = parser.parse_args()
    args.model = "meta-llama/Llama-3.2-1B-Instruct"
    args.task = "arc-oai"

    # Set the device
    device = torch.device(args.device)

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Remove the today's date from the prompt for reproducibility
    if args.model in ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]:
        tokenizer.chat_template = LLAMA_CHAT_TEMPLATE

    # Load the dataset
    dataset = load_task_dataset(args.task)

    # Set the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # Create the dataset constructor
    steering_config = SteeringConfig(
        tokenizer=tokenizer,
        encoding_method="instruct",
        add_question=True,
        num_fewshot_examples=10,
        n_contrastive_samples=200,
        add_generation_prompt=True,
        sample_selection_method="distance",
        append_special_token=True,
        c_keys=0.0,
        c_values=6.0,
        layers_ids_keys=[1],                # Apply to all layers
        layers_ids_values=[1],              # Apply to all layers
    )

    # Create the dataset constructor
    dataset_constructor = ContrastiveDatasetConstructor(
        dataset=dataset["train"],
        steering_config=steering_config,
        task=args.task,
    )
    contrastive_data = dataset_constructor.construct_dataset()

    # Create the evaluation dataset constructor
    eval_constructor = EvaluationDatasetConstructor(
        dataset=dataset["test"],
        tokenizer=tokenizer,
        n=args.n,
        num_fewshot_prompt=0,
        task=args.task,
        prefix=None,
        system_prompt=None,
        encoding_method="instruct",
        add_generation_prompt=True,
    )
    evaluation_dataset = eval_constructor.construct_dataset()

    training_samples = [
        (
            contrastive_data[i]['positive'],
            contrastive_data[i]['negative']
        )
        for i in range(len(contrastive_data))
    ]

    # Construct the activations steering vector
    steering_vector = train_steering_vector(model, tokenizer, training_samples, layers=[7], show_progress=True)

    # Construct the cache steering vector
    steering_kv = extract_steering_kv(
        model,
        tokenizer,
        contrastive_data,
        steering_config,
        device=device,
    )

    # Set generation arguments
    generation_kwargs = {"do_sample": False, "max_new_tokens": 256}

    # Load the already existing results
    results_path = os.path.join(args.output_dir, "all_results.json")
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for run_id in range(args.n_runs):
        run_tag = f"run_{run_id}"
        for batch_size in [1, 16]:

            # Baseline
            baseline_key = f"baseline_{batch_size}"
            if baseline_key in all_results and run_tag in all_results[baseline_key] and not args.rerun_existing:
                logger.info(f"Skipping {baseline_key} {run_tag}: results already exist.")
            else:
                baseline_results = generate_baseline(
                    model,
                    tokenizer,
                    evaluation_dataset,
                    device=device,
                    batch_size=batch_size,
                    generation_kwargs=copy(generation_kwargs),
                )
                all_results.setdefault(baseline_key, {})[run_tag] = {
                    "total_generated_tokens": sum(baseline_results["n_generated_tokens"]),
                    "total_generation_time": sum(baseline_results["generation_times"]),
                    "average_generation_time": np.mean(baseline_results["generation_times"]),
                    "average_generated_tokens": np.mean(baseline_results["n_generated_tokens"]),
                    "time_per_token": sum(baseline_results["generation_times"]) / sum(baseline_results["n_generated_tokens"]),
                    "batch_size": batch_size,
                    "name": "baseline",
                }
                save_results(results_path, all_results)
                logger.info(f"{baseline_key} {run_tag} results saved.")

            # Cache steering
            cache_key = f"cache_steering_{batch_size}"
            if cache_key in all_results and run_tag in all_results[cache_key] and not args.rerun_existing:
                logger.info(f"Skipping {cache_key} {run_tag}: results already exist.")
            else:
                cache_steering_results = generate_cache_steering(
                    model,
                    tokenizer,
                    evaluation_dataset,
                    device=device,
                    batch_size=batch_size,
                    generation_kwargs=copy(generation_kwargs),
                    steering_kv=steering_kv,
                    steering_config=steering_config
                )
                all_results.setdefault(cache_key, {})[run_tag] = {
                    "total_generated_tokens": sum(cache_steering_results["n_generated_tokens"]),
                    "total_generation_time": sum(cache_steering_results["generation_times"]),
                    "average_generation_time": np.mean(cache_steering_results["generation_times"]),
                    "average_generated_tokens": np.mean(cache_steering_results["n_generated_tokens"]),
                    "time_per_token": sum(cache_steering_results["generation_times"]) / sum(cache_steering_results["n_generated_tokens"]),
                    "batch_size": batch_size,
                    "name": "cache_steering",
                }
                save_results(results_path, all_results)
                logger.info(f"{cache_key} {run_tag} results saved.")

            # Activation steering
            for continuous in [True, False]:
                activation_key = f"activation_steering_{batch_size}_{continuous}"
                if activation_key in all_results and run_tag in all_results[activation_key] and not args.rerun_existing:
                    logger.info(f"Skipping {activation_key} {run_tag}: results already exist.")
                else:
                    activation_steering_results = generate_activation_steering(
                        model,
                        tokenizer,
                        evaluation_dataset,
                        device=device,
                        batch_size=batch_size,
                        generation_kwargs=copy(generation_kwargs),
                        steering_vector=steering_vector,
                        multiplier=1.0 if continuous else 5.0,
                        continuous=continuous
                    )
                    all_results.setdefault(activation_key, {})[run_tag] = {
                        "total_generated_tokens": sum(activation_steering_results["n_generated_tokens"]),
                        "total_generation_time": sum(activation_steering_results["generation_times"]),
                        "average_generation_time": np.mean(activation_steering_results["generation_times"]),
                        "average_generated_tokens": np.mean(activation_steering_results["n_generated_tokens"]),
                        "time_per_token": sum(activation_steering_results["generation_times"]) / sum(activation_steering_results["n_generated_tokens"]),
                        "batch_size": batch_size,
                        "continuous": continuous,
                        "name": "activation_steering",
                    }
                    save_results(results_path, all_results)
                    logger.info(f"{activation_key} {run_tag} results saved.")
