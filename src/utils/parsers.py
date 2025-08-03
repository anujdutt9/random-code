from argparse import ArgumentParser, BooleanOptionalAction

from src.utils.constants import AggregationMethods


def pairs_construction_args(parser=None):
    """
    Adds arguments related to pairs construction.
    """
    if parser is None:
        parser = ArgumentParser(description="Pairs construction arguments parser")
    
    parser.add_argument("--num_fewshot_examples", type=int, default=0, help="Number of few-shot examples in prompts for steering vector extraction")
    parser.add_argument("--n_contrastive_samples", type=int, default=None, help="Number of examples for steering vector extraction")
    parser.add_argument("--prefix", type=str, default="Let's think step by step.", help="Prefix to add to prompt for steering vector extraction")
    parser.add_argument("--add_question", default=False, action=BooleanOptionalAction, help="Include the question to prompt for steering vector extraction")
    parser.add_argument("--add_prefix", default=False, action=BooleanOptionalAction, help="Whether to add prefix to prompt for steering vector extraction")
    parser.add_argument("--add_answer", default=False, action=BooleanOptionalAction, help="Include the answer to prompt for steering vector extraction")
    parser.add_argument("--sample_fewshot_examples", default=True, action=BooleanOptionalAction, help="Whether to sample few-shot examples from training set vs. using the pre-defined ones")
    parser.add_argument("--fewshot_delimeter", type=str, default="\n\n", help="Delimiter to separate in-context examples with")
    parser.add_argument("--fewshot_only_in_postive", default=False, action=BooleanOptionalAction, help="Whether to include only the in-context examples in the negative examples")
    parser.add_argument("--use_tokenized_version", default=False, action=BooleanOptionalAction, help="Whether to use the tokenized version of the prompt for steering vector extraction")
    parser.add_argument("--encoding_method", type=str, default="qa", choices=["qa", "instruct"], help="Type of encoding to use for the prompt")
    parser.add_argument("--extraction_system_prompt", type=str, default=None, help="System prompt to include in the prompt")
    parser.add_argument("--add_generation_prompt", default=False, action=BooleanOptionalAction, help="Whether to add generation prompt to the prompt both during extraction and application")
    parser.add_argument("--sample_selection_method", type=str, default="random", choices=["random", "distance"], help="Method to select examples for steering vector extraction: 'random' selects random examples, 'distance' selects examples based on distances")
    
    return parser


def steering_extraction_args(parser=None):
    """
    Adds arguments related to extracting steering vectors.
    """
    if parser is None:
        parser = ArgumentParser(description="Steering extraction arguments parser")
    
    parser.add_argument("--extraction_method", type=str, default="last_token", choices=["sequence_mean", "last_token", "prefix_mean", "penultimate_token"], help="Method to extract steering vectors: 'sequence_mean' takes mean of all tokens, 'last_token' takes last token, 'prefix_mean' takes mean of the prefix")
    parser.add_argument("--take_difference", default=True, action=BooleanOptionalAction, help="Whether to take the difference between the steering vectors")
    parser.add_argument("--aggregation_method", type=str, default="mean", choices=AggregationMethods.values(), help="Method to aggregate steering vectors: 'mean' takes mean of all vectors, 'clustering' clusters the vectors and takes the mean of each cluster, 'none' does not aggregate")
    parser.add_argument("--cluster_layer_id", type=int, default=None, help="Layer to cluster on")
    parser.add_argument("--clustering_method", type=str, default="kmeans", choices=["kmeans"], help="Clustering method to use")
    parser.add_argument("--cluster_on", type=str, default=None, choices=["activations", "keys", "values", "keys+values"], help="Entity to cluster on: 'activations', 'keys', 'values', or 'keys+values'")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters to use for clustering")
    return parser


def applying_steering_args(parser=None):
    """
    Adds arguments related to applying steering vectors.
    """
    if parser is None:
        parser = ArgumentParser(description="Applying steering arguments parser")
    
    parser.add_argument("--how", type=str, default="last", help="Apply steering to 'all' tokens or 'last' token only")
    parser.add_argument("--append_special_token", default=False, action=BooleanOptionalAction, help="Whether to append a special token to the input before applying the the steering vectors")
    
    return parser


def prompt_construction_args(parser=None):
    """
    Adds arguments related to prompt construction.
    """
    if parser is None:
        parser = ArgumentParser(description="Prompt construction arguments parser")
    
    parser.add_argument("--num_fewshot_prompt", type=int, default=0, help="Number of few-shot examples to include in prompt")
    parser.add_argument("--append_prefix_to_prompt", default=False, action=BooleanOptionalAction, help="Whether to append prefix to the prompt")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to include in the prompt")
    
    return parser


def cache_steering_args(parser=None):
    """
    Adds arguments related to cache steering.
    """
    if parser is None:
        parser = ArgumentParser(description="Cache steering arguments parser")

    parser.add_argument("--c_keys", type=float, default=None, help="Weight to apply to the keys in the cache")
    parser.add_argument("--c_values", type=float, default=None, help="Weight to apply to the values in the cache")
    parser.add_argument("--layers_ids_keys", type=int, nargs="+", default=None, help="Layers to apply steering to the keys in the cache")
    parser.add_argument("--layers_ids_values", type=int, nargs="+", default=None, help="Layers to apply steering to the values in the cache")

    return parser
