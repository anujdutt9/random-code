from argparse import ArgumentParser, BooleanOptionalAction
from copy import deepcopy
import random
import uuid
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from steering_vectors import train_steering_vector

from src.evaluation.evaluator import Evaluator
from src.data.loader import load_task_dataset
from src.data.contrastive_dataset import ContrastiveDatasetConstructor
from src.data.evaluation_dataset import EvaluationDatasetConstructor
from src.steering.config import SteeringConfig
from src.utils.logging_setup import logger, log_stream
from src.utils.constants import LLAMA_CHAT_TEMPLATE
from src.utils.parsers import pairs_construction_args
from src.utils.helpers import save_experiment_results


def select_layers(activations, layers):
    if layers == "all":
        return activations
    return {layer: activations[layer] for layer in layers}

# Try only the middle layers as per the different paperes
LAYERS_MAP = {
    "HuggingFaceTB/SmolLM2-360M-Instruct": [13, 14, 15, 16, 17, 18, 19],
    "meta-llama/Llama-3.2-1B-Instruct": [6, 7, 8, 9, 10],
    "meta-llama/Llama-3.2-3B-Instruct": [13, 14, 15],
    "mistralai/Mistral-7B-Instruct-v0.3": [13, 14, 15, 16, 17, 18, 19],
    "Qwen/Qwen2-0.5B-Instruct": [11, 12, 13],
    "meta-llama/Llama-3.1-8B-Instruct": [15, 16, 17],
    "microsoft/Phi-4-mini-instruct": [15, 16, 17],
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--task', type=str)

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default="activation_steering_results")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--verbose', default=True, action=BooleanOptionalAction)
    parser.add_argument('--n', type=int, default=None)

    # Multipliers for the search
    parser.add_argument('--multipliers', type=float, nargs='+', default=[0.5, 1, 3], help="Multipliers for steering vector")
    parser.add_argument('--layers', type=int, nargs='+', default=None, help="Layers to apply steering to")

    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--force_choice", default=True, action=BooleanOptionalAction)
    parser.add_argument("--skip_existing", default=False, action=BooleanOptionalAction)

    pairs_construction_args(parser)

    args = parser.parse_args()

    # Set the device
    device = torch.device(args.device)

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Remove the today's date from the prompt for reproducibility
    if args.model in ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]:
        tokenizer.chat_template = LLAMA_CHAT_TEMPLATE

    # Load the task dataset
    dataset = load_task_dataset(args.task)

    # Construct the evaluation dataset
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

    if args.verbose:
        logger.info(f"First example in the original evaluation subset: {dataset['test'][0]}")
        logger.info(f"First example in evaluation dataset after preprocessing: {evaluation_dataset[0]}")

    # Construct the contrastive dataset
    steering_config = SteeringConfig(tokenizer=tokenizer, **vars(args))
    contrastive_constructor = ContrastiveDatasetConstructor(
        dataset=dataset["train"],
        steering_config=steering_config,
        task=args.task
    )
    contrastive_dataset = contrastive_constructor.construct_dataset()

    training_samples = [
        (
            contrastive_dataset[i]['positive'],
            contrastive_dataset[i]['negative']
        )
        for i in range(len(contrastive_dataset))
    ]

    # Extract the steering vector for all layers
    steering_vector = train_steering_vector(
        model,
        tokenizer,
        training_samples,
        show_progress=True,
        batch_size=args.batch_size,
    )
    # Save the activations separately to select the layers
    all_activations = deepcopy(steering_vector.layer_activations)

    # Load all previous experiments
    previous_experiments = []
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    arguments = data["arguments"]
                    if arguments["task"] == args.task and arguments["model"] == args.model:
                        previous_experiments.append(arguments)
                        logger.info(f"Found previous experiment: {file}")
    previous_experiments = pd.DataFrame(previous_experiments)

    # Define the generation arguments for experiments
    max_new_tokens = args.max_new_tokens
    generation_kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens}
    logger.info(f"Experiment generation arguments: {generation_kwargs}")

    all_layers = (
        [[l] for l in args.layers]
        if args.layers
        else [[l] for l in LAYERS_MAP.get(args.model, [])]
    )

    # Run a grid search over the steering strenghts multipliers and layers
    for multiplier in args.multipliers:
        for layers in all_layers:

            # Check if the experiment has already been run
            if not previous_experiments.empty and previous_experiments[
                (previous_experiments["multiplier"] == multiplier) &
                (previous_experiments["layers"].apply(lambda x: str(x) == str(layers)))
            ].shape[0] > 0 and not args.skip_existing:
                logger.info(f"Experiment with multiplier {multiplier} and layers {layers} already run. Skipping.")
                continue

            experiment_id = str(uuid.uuid4())
            logger.info(f"Experiment ID: {experiment_id}")
            logger.info(f"Applying steering vector with multiplier {multiplier} to layers {layers}")

            # Set seeds for reproducibility
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            random.seed(args.seed)

            # Select the layers to apply steering to
            selected_activations = select_layers(all_activations, layers)
            steering_vector.layer_activations = selected_activations
            steering_vector = steering_vector.to(device=device)

            # Continous application of the steering vector (as per CoT paper by Jason Zhang and Meta-Analysis)
            try:
                # Patch the model with the steering vector
                handle = steering_vector.patch_activations(
                    model=model,
                    multiplier=multiplier,
                    min_token_index=-1, # Apply to the last token only
                )

                # Save argumetns to dict
                args_dict = vars(args).copy()
                args_dict.update(generation_kwargs)
                args_dict["layers"] = str(layers)
                args_dict["multiplier"] = multiplier
                args_dict["num_fewshot_prompt"] = 0

                # Run the baseline evaluation
                evaluator = Evaluator(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=evaluation_dataset,
                    task=args.task,
                    device=args.device,
                    encoding_method=args.encoding_method,
                    force_choice=args.force_choice,
                )
                result = evaluator.evaluate(
                    generation_kwargs=deepcopy(generation_kwargs),
                    batch_size=args.batch_size,
                    use_cache=False, # DO NOT use cache with continuous application
                )
                save_experiment_results(result, args_dict, experiment_id, args.task, is_baseline=False)

            finally:
                handle.remove()
