from collections import defaultdict

from tqdm.auto import tqdm
from transformers import DynamicCache, BatchEncoding, PreTrainedModel
import torch

from src.steering.config import SteeringConfig
from src.utils.constants import AggregationMethods
from src.utils.helpers import pad_tokens, get_token_to_append
from src.utils.clustering import cluster_kv_vectors
from src.utils.logging_setup import logger


def extract_steering_kv(
    model,
    tokenizer,
    data,
    steering_config: SteeringConfig,
    batch_size=1,
    device="cpu",
):

    steering_values = defaultdict(lambda: torch.tensor([]).to(device))
    steering_keys = defaultdict(lambda: torch.tensor([]).to(device))
    activations = defaultdict(lambda: torch.tensor([]))
    output_dict = {}

    for example in tqdm(data.iter(batch_size=batch_size)):

        # Tokenize the data
        if steering_config.use_tokenized_version:
            pos_tokens = pad_tokens(example['tokenized_positive'], padding_side=steering_config.padding_side, device=device)
            neg_tokens = pad_tokens(example['tokenized_negative'], padding_side=steering_config.padding_side, device=device)
        else:
            pos_tokens = tokenizer(example['positive'], return_tensors='pt', padding=True).to(device)
            neg_tokens = tokenizer(example['negative'], return_tensors='pt', padding=True).to(device)

        # Select the indices of the last token in the sequence
        if steering_config.extraction_method == "last_token":
            subtract_index = 1
        elif steering_config.extraction_method == "penultimate_token":
            subtract_index = 2
        elif isinstance(steering_config.extraction_method, int):
            subtract_index = steering_config.extraction_method
        else:
            raise ValueError(f"Invalid value provided for extraction_method: {steering_config.extraction_method}. Valid values are ['last_token', 'penultimate_token'] or integers.")

        # Find indices of the tokens to extract the steering vectors from
        pos_indices = pos_tokens['attention_mask'].sum(dim=1) - subtract_index
        neg_indices = neg_tokens['attention_mask'].sum(dim=1) - subtract_index
        batch_indices = torch.arange(pos_tokens['input_ids'].size(0), device=pos_tokens['input_ids'].device)

        # Log the tokens that are used to extract the steering vectors
        if steering_config.verbose:
            extraction_token_pos = [tokenizer.decode(i) for i in pos_tokens['input_ids'][batch_indices, pos_indices].tolist()]
            extraction_token_neg = [tokenizer.decode(i) for i in neg_tokens['input_ids'][batch_indices, neg_indices].tolist()]
            logger.debug(f"Extracting steering vectors for the tokens '{extraction_token_pos}' and '{extraction_token_neg}'")

        # Run the model with the hook to cache activations
        cache_positive, cache_negative = DynamicCache(), DynamicCache()
        with torch.no_grad():
            pos_out = model(**pos_tokens, output_hidden_states=True, past_key_values=cache_positive)
            neg_out = model(**neg_tokens, output_hidden_states=True, past_key_values=cache_negative)

        for layer_id in range(len(cache_positive.value_cache)):
            pos_values = cache_positive.value_cache[layer_id][batch_indices, :, pos_indices, :]
            neg_values = cache_negative.value_cache[layer_id][batch_indices, :, neg_indices, :]
            pos_activations = pos_out.hidden_states[layer_id+1][batch_indices, pos_indices, :].cpu()

            pos_keys = cache_positive.key_cache[layer_id][batch_indices, :, pos_indices, :]
            neg_keys = cache_negative.key_cache[layer_id][batch_indices, :, neg_indices, :]
            neg_activations = neg_out.hidden_states[layer_id+1][batch_indices, neg_indices, :].cpu()

            # Take the differnece between the vectors
            if steering_config.take_difference:
                steering_values[layer_id] = torch.cat([steering_values[layer_id], pos_values - neg_values]) # [batch_size, n_heads, head_dim]
                steering_keys[layer_id] = torch.cat([steering_keys[layer_id], pos_keys - neg_keys])
                activations[layer_id] = torch.cat([activations[layer_id], pos_activations - neg_activations])
            else:
                steering_values[layer_id] = torch.cat([steering_values[layer_id], pos_values], dim=0) # [batch_size, n_heads, head_dim]
                steering_keys[layer_id] = torch.cat([steering_keys[layer_id], pos_keys], dim=0)
                activations[layer_id] = torch.cat([activations[layer_id], pos_activations], dim=0)

    if steering_config.aggregation_method == AggregationMethods.mean:
        for layer_id in steering_values:
            steering_values[layer_id] = torch.mean(steering_values[layer_id], dim=0) # [n_heads, head_dim]
            steering_keys[layer_id] = torch.mean(steering_keys[layer_id], dim=0)

        output_dict["values"] = dict(steering_values)
        output_dict["keys"] = dict(steering_keys)

    elif steering_config.aggregation_method == AggregationMethods.clustering:
        steering_keys, steering_values, labels = cluster_kv_vectors(
            activations,
            steering_keys,
            steering_values,
            steering_config.cluster_layer_id,
            steering_config.n_clusters,
            steering_config.seed,
            method=steering_config.clustering_method,
            cluster_on=steering_config.cluster_on,
        )
        output_dict["values"] = steering_values
        output_dict["keys"] = steering_keys
        output_dict["labels"] = labels
    
    output_dict["activations"] = dict(activations)

    return output_dict


def generate_with_cache_steering(
    model: PreTrainedModel,
    tokens,
    steering_kv,
    steering_config: SteeringConfig,
    output_full_dict=False,
    **kwargs,
):

    # Check the format of tokens
    if isinstance(tokens, BatchEncoding):
        tokens = tokens["input_ids"]

    # Check if the task was provided
    task = None
    if "task" in kwargs:
        task = kwargs.pop("task")

    if steering_config.how == "last":
        application_token_idx = -1
    elif isinstance(steering_config.how, int):
        application_token_idx = steering_config.how
    else:
        raise ValueError(f"Invalid value provided for how: {steering_config.how}. Valid values are ['last'] or integers.")

    # Append a special token to the input tokens if needed to be able to steer the cache of last token
    if steering_config.append_special_token and application_token_idx == -1:
        token_to_append = get_token_to_append(steering_config, tokens, task=task)
        logger.debug(f"Appending special token '{steering_config.tokenizer.decode(token_to_append[0].item())}' to the input tokens.")
        tokens = torch.cat([tokens, token_to_append], dim=-1)
        if "attention_mask" in kwargs:
            kwargs['attention_mask'] = torch.cat([kwargs['attention_mask'], torch.ones_like(token_to_append)], dim=-1)

    # Log the tokens that the steering is applied to
    if steering_config.verbose:
        decoded_last_tokens = [steering_config.tokenizer.decode(i) for i in tokens[:, application_token_idx-1].tolist()]
        logger.debug(f"Applying steering to the cache of the following tokens '{decoded_last_tokens}'")

    # Create the initial cache
    cache_input = {
        "input_ids": tokens,
        "attention_mask": kwargs['attention_mask']
    }
    past_key_values = precompute_kv_cache(model, cache_input)

    # Steer the cache
    past_key_values = steer_kv_cache(
        past_key_values,
        steering_kv,
        steering_config,
        application_token_idx=application_token_idx,
    )

    try:
        # Generate
        output = model.generate(
            tokens,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            **kwargs,
        )
        if output_full_dict:
            return output

        output_tokens = output.sequences

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise e

    return output_tokens


def precompute_kv_cache(model, tokens):
    """
    Precompute the key and value caches for the input tokens except the last one.
    """
    past_key_values = DynamicCache()

    if isinstance(tokens, BatchEncoding) or isinstance(tokens, dict):    
        cache_input = {
            k: v[:, :-1]
            for k, v in tokens.items()
            if k in ["input_ids", "attention_mask", "token_type_ids", "position_ids"]
        }
    else:
        cache_input = {
            "input_ids": tokens[:, :-1],
        }

    # Compute correct position_ids before caching
    seq_lengths = cache_input["attention_mask"].sum(dim=1)
    position_ids = torch.zeros_like(cache_input["input_ids"])
    for i in range(cache_input["input_ids"].shape[0]):
        valid_len = seq_lengths[i]
        position_ids[i, -valid_len:] = torch.arange(valid_len)
    cache_input["position_ids"] = position_ids

    with torch.no_grad():
        model(**cache_input, past_key_values=past_key_values, use_cache=True)

    return past_key_values


def steer_kv_cache(cache, steering_kv, steering_config, application_token_idx=-1):
    
    if "values" in steering_kv:
        # Steer the values cache
        for layer_idx, past_values in steering_kv["values"].items():
            steer_kv_cache_layer(cache, past_values, steering_config, layer_idx, type='values', application_token_idx=application_token_idx)

    if "keys" in steering_kv:
        # Steer the keys cache
        for layer_idx, past_keys in steering_kv["keys"].items():
            steer_kv_cache_layer(cache, past_keys, steering_config, layer_idx, type='keys', application_token_idx=application_token_idx)

    return cache


def steer_kv_cache_layer(cache, steering_vector, steering_config, layer_idx, type='values', application_token_idx=-1):
    """
    Steer the key and value cache of a specific layer.
    """
    sv = steering_vector.clone() # [n_heads, head_dim]

    # Apply the vector to the cache
    if type == 'values':
        cache.value_cache[layer_idx][:, :, application_token_idx, :] += sv * steering_config.c_values
    elif type == 'keys':
        cache.key_cache[layer_idx][:, :, application_token_idx, :] += sv * steering_config.c_keys


def prune_steering_heads(steering_kv, top_k=4):
    """
    Zeroes out all but the top-k most important heads (per layer), based on value vector norm.
    """
    for layer_idx in steering_kv["values"]:
        v = steering_kv["values"][layer_idx]  # [n_heads, head_dim]
        k = steering_kv["keys"][layer_idx]    # [n_heads, head_dim]

        # Compute importance per head
        head_importance = v.norm(dim=-1)  # shape: [n_heads]

        # Select top-k heads
        topk_indices = torch.topk(head_importance, top_k).indices

        # Zero out other heads
        mask = torch.zeros_like(head_importance, dtype=torch.bool)
        mask[topk_indices] = True

        steering_kv["values"][layer_idx] = v * mask.unsqueeze(-1)
        steering_kv["keys"][layer_idx] = k * mask.unsqueeze(-1)

    return steering_kv
