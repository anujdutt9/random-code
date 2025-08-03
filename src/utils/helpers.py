"""
Helper functions for the project.
"""

from datetime import datetime
import json
import logging
import os
import uuid
import re

from transformers import LogitsProcessor
import torch

from src.steering.config import SteeringConfig
from src.utils.constants import VECTOR_AFFECTING_PARAMETERS, Tasks
from src.utils.logging_setup import logger


class ForceChoiceProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.allowed] = scores[:, self.allowed]
        return mask


def save_experiment_results(results, args_dict, experiment_id, task, is_baseline=False):
    """
    Save the results of an experiment
    """
    
    # Set directory and file name
    experiment_name = f"{args_dict['model'].split('/')[-1]}_{task}_{args_dict['num_fewshot_prompt']}"
    if is_baseline:
        experiment_name += "_baseline"
    target_dir = os.path.join(args_dict['output_dir'], experiment_name)

    # Add metadata
    args_dict["timestamp"] = str(datetime.now())
    args_dict["experiment_id"] = experiment_id

    # Drop the unnecessary arguments
    del args_dict["output_dir"]
    if "verbose" in args_dict:
        del args_dict["verbose"]

    # Create the final results dict
    final_results = {
        "arguments": args_dict,
    }
    final_results.update(results)

    # Save the file
    os.makedirs(target_dir, exist_ok=True)
    results_file_name = os.path.join(target_dir, datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S_%f") + ".json")
    logger.info(f"Saving results to: {results_file_name}")
    with open(results_file_name, "w") as f:
        json.dump(final_results, f, indent=4)


def pad_tokens(tokens, padding_side, device):
    # Find the longest sequence
    max_length = max(len(token) for token in tokens)

    # Pad the tokens
    padded_tokens = []
    attention_masks = []
    for token in tokens:
        if padding_side == "left":
            padded_tokens.append([0] * (max_length - len(token)) + token)
            attention_masks.append([0] * (max_length - len(token)) + [1] * len(token))
        else:
            padded_tokens.append(token + [0] * (max_length - len(token)))
            attention_masks.append([1] * len(token) + [0] * (max_length - len(token)))

    # Convert to tensor
    return {"input_ids": torch.tensor(padded_tokens, device=device), "attention_mask": torch.tensor(attention_masks, device=device)}


def get_token_to_append(steering_config: SteeringConfig, tokens, task=None):
    tokenizer = steering_config.tokenizer

    if steering_config.add_generation_prompt:
        if task in [Tasks.gsm8k_oai, Tasks.gsm8k]:
            token = tokenizer(" ", add_special_tokens=False)["input_ids"][0]
        else:
            token = tokenizer("\n", add_special_tokens=False)["input_ids"][0]

    else:
        if tokenizer.name_or_path in ["HuggingFaceTB/SmolLM2-360M-Instruct"]:
            token = tokenizer.bos_token_id

        elif tokenizer.name_or_path in [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        ]:
            token = 128006

    return torch.ones(tokens.shape[0], 1, device=tokens.device, dtype=tokens.dtype) * token


def select_steering_kv_layers(steering_cache, steering_config: SteeringConfig):
    # Select the correct layers
    if steering_config.layers_ids_keys is not None:
        if len(steering_config.layers_ids_keys) > 1:
            logging.info(f"Selecting layers {steering_config.layers_ids_keys} for keys.")
            steering_cache['keys'] = {k: v for k, v in steering_cache['keys'].items() if k in steering_config.layers_ids_keys}
        else:
            logging.info(f"Selecting layers from {steering_config.layers_ids_keys} and above for keys.")
            steering_cache['keys'] = {k: v for k, v in steering_cache['keys'].items() if k >= steering_config.layers_ids_keys[0]}

    if steering_config.layers_ids_values is not None:
        if len(steering_config.layers_ids_values) > 1:
            logging.info(f"Selecting layers {steering_config.layers_ids_values} for values.")
            steering_cache['values'] = {k: v for k, v in steering_cache['values'].items() if k in steering_config.layers_ids_values}
        else:
            logging.info(f"Selecting layers from {steering_config.layers_ids_values} and above for values.")
            steering_cache['values'] = {k: v for k, v in steering_cache['values'].items() if k >= steering_config.layers_ids_values[0]}

    return steering_cache


def generate_vector_id(steering_config: SteeringConfig, model_name: str, task: str, subtask: str = None):
    """
    Generate a unique ID for a steering vector based on its properties
    """
    properties = []
    for property in VECTOR_AFFECTING_PARAMETERS:
        properties.append(str(getattr(steering_config, property)))

    concatenated = "|".join(map(str, properties))
    concatenated += f"|{model_name}"
    concatenated += f"|{task}"
    if subtask:
        concatenated += f"|{subtask}"
    unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, concatenated))
    logger.info(f"Generated vector ID: {unique_id}")
    return unique_id


def load_vector(vector_id: str, cache_dir: str, device="cuda"):
    """
    Load a steering vector from a file
    """
    logger.info(f"Loading vector from {cache_dir}/{vector_id}.pt")
    return torch.load(f"{cache_dir}/{vector_id}.pt", map_location=device)


def save_vector(vector: torch.Tensor, vector_id: str, cache_dir: str):
    """
    Save a steering vector to a file
    """
    logger.info(f"Saving vector to {cache_dir}/{vector_id}.pt")
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(vector, f"{cache_dir}/{vector_id}.pt")


def compute_example_length(example):
    """
    Compute the length of an example.
    """

    example['length'] = len(example['input'])
    return example


def get_gen_kwargs(model):
    """
    Get the generation kwargs from the model's generation config.
    If `do_sample` is set to True, use the values from the config.
    Otherwise, use default values: 1.0 for repetition penalty, 0.9 
    for top_p, and 50 for top_k.
    
    Args:
        model: The model to get the generation config from.
        
    Returns:
        dict: The generation kwargs.
    """
    gen_config = vars(model.generation_config)

    if gen_config['do_sample']:
        gen_kwargs = {
            "do_sample": True,
            "temperature": gen_config['temperature'],
            "top_p": gen_config['top_p'],
            "top_k": gen_config['top_k'],
        }
    else:
        gen_kwargs = {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 50,
        }
    return gen_kwargs


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def strip_chat_template(text):
    # Remove common model-specific tags
    text = re.sub(r"</?s>", "", text)
    text = re.sub(r"\[/?INST\]", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<\|start_header_id\|>assistant<\|end_header_id\|>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<\|.*?\|>", "", text)  # Matches <|...|>
    # Remove "Let's think step by step" (and minor variants, case-insensitive)
    text = re.sub(r"let'?s think step by step[.:]?", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_choices(text):
    # Find the "Choices:" section
    match = re.search(r"Choices:\s*(.*)", text, re.DOTALL)
    if not match:
        return {}

    choices_text = match.group(1).strip()

    # Extract choices using regex
    pattern = re.findall(r"([A-E])\s*:\s*(.*?)(?:\n|$)", choices_text)

    # Clean trailing punctuation
    return {
        label: remove_leading_article(choice.strip().rstrip("."))
        for label, choice in pattern
    }


def remove_leading_article(phrase):
    return re.sub(r"^(a|an|the)\s+", "", phrase.strip(), flags=re.IGNORECASE)


def get_metric_key(task_name):
    return 'last_digit' if task_name in ['gsm8k-oai', 'gsm8k'] else 'augmented_extract'


def load_vectors_and_activations(
    vector_idx,
    task_name,
    model_name_split,
    train_activations_bert,
    test_activations_bert,
    layer_idx=8,
    n_test_examples=160,
):
    """
    Load steering vectors and activations for the test and train sets, and organize them into a dictionary.

    Args:
        vector_idx (str): The vector index to load steering vectors.
        task_name (str): The name of the task.
        model_name_split (str): The model name split.
        train_activations_bert (torch.Tensor): Precomputed BERT embeddings for the train set.
        test_activations_bert (torch.Tensor): Precomputed BERT embeddings for the test set.
        layer_idx (int): The layer index to select activations from. Default is 8.
        n_examples (int): The number of examples to consider. Default is 160.

    Returns:
        dict: A dictionary containing train and test activations organized by type.
        torch.Tensor: Labels from the steering vectors.
    """
    # Load steering vectors
    steering_kv = load_vector(vector_idx, "cached_vectors", "cpu")

    # Load activations from the model
    test_activations = load_vector(f"{task_name}_0_test_{model_name_split}_activations", "artifacts", "cpu")
    train_activations = load_vector(f"{task_name}_0_train_{model_name_split}_activations", "artifacts", "cpu")
    n = 5 if task_name in ["gsm8k", "gsm8k-oai"] else 10
    train_activations_pos = load_vector(f"{task_name}_{n}_train_{model_name_split}_activations", "artifacts", "cpu")

    # Organize activations into a dictionary
    n_train_examples = steering_kv['labels'].shape[0]
    vector_options = {
        "train": {
            "icl_question_diff_activations": steering_kv['activations'][layer_idx][:n_train_examples],
            "icl_question_pos_activations": train_activations_pos['activations'][layer_idx][:n_train_examples],
            "question_activations": train_activations['activations'][layer_idx][:n_train_examples],
            "question_activations_bert": train_activations_bert[:n_train_examples],
        },
        "test": {
            "question_activations": test_activations['activations'][layer_idx][:n_test_examples],
            "question_activations_bert": test_activations_bert[:n_test_examples],
        }
    }

    return vector_options, steering_kv['labels']


def sort_clusters(correct_answers):
    """
    Sort the correct answers according to the cluster ids.
    """
    # Order the correct answers according to the cluster ids
    n_clusters = int(max(correct_answers.keys()) + 1)
    correct_answers_list = []
    for i in range(n_clusters):
        correct_answers_list.append(correct_answers[i])
    correct_answers_tensor = torch.tensor(correct_answers_list)
    return correct_answers_tensor, n_clusters
