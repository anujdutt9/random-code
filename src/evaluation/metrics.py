from collections import defaultdict

import numpy as np

from src.evaluation.filters import FILTERS
from src.utils.constants import Tasks


def extract_answer(text: str, task: str, question: str = None):
    """
    Extract the final answer from model output.

    Args:
        text (str): Model output text
        task (str): Type of dataset to determine extraction method
        question (str, optional): The question asked to the model. Defaults to None.

    Returns:
        str: Extracted answer
    """
    return {filter.name: filter(text, question) for filter in FILTERS[task]}


def compute_metrics(results, task: str):
    """
    Calculate evaluation metrics.

    Args:
        results (list): List of evaluation results
        task (str): Type of dataset for answer extraction

    Returns:
        dict: Metrics
    """
    for result in results:
        question = result["input"] if task in [Tasks.gsm8k_oai, Tasks.gsm8k] else None
        result["filtered_response"] = extract_answer(result["response"], task, question)

    # Calculate accuracy
    metrics = compute_per_filter_metrics(results)
    return metrics


def compute_per_filter_metrics(results):
    """
    Calculate evaluation metrics for each filter.

    Args:
        results (list): List of evaluation results

    Returns:
        dict: Metrics
    """
    per_filter_results = defaultdict(list)
    for result in results:
        for filter_name, filtered_text in result["filtered_response"].items():
            per_filter_results[filter_name].append(
                1 if filtered_text == result["target"] else 0
            )

    return {k: np.mean(v) for k, v in per_filter_results.items()}
