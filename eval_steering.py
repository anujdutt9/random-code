from argparse import ArgumentParser, BooleanOptionalAction
from copy import deepcopy
import uuid
import random
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

from src.data.loader import load_task_dataset
from src.data.evaluation_dataset import EvaluationDatasetConstructor
from src.data.contrastive_dataset import ContrastiveDatasetConstructor
from src.evaluation.evaluator import Evaluator
from src.utils.constants import Tasks, EncodingMethods, LLAMA_CHAT_TEMPLATE
from src.utils.parsers import (
    prompt_construction_args,
    pairs_construction_args,
    steering_extraction_args,
    applying_steering_args,
    cache_steering_args,
)
from src.utils.helpers import save_experiment_results, get_gen_kwargs
from src.steering.config import SteeringConfig
from src.utils.logging_setup import logger, log_stream

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
logging_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logging.conf')
# logging.config.fileConfig(logging_config_path)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--subtask", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--extraction_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--verbose", default=True, action=BooleanOptionalAction)
    parser.add_argument("--vector_cache_dir", type=str, default="cached_vectors")

    parser.add_argument("--temperatures", nargs="*", default=list(), type=float)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--eval_type", type=str, default="greedy")
    parser.add_argument("--force_choice", default=True, action=BooleanOptionalAction)

    parser = pairs_construction_args(parser)
    parser = steering_extraction_args(parser)
    parser = applying_steering_args(parser)
    parser = cache_steering_args(parser)
    parser = prompt_construction_args(parser)

    args = parser.parse_args()
    # print(f"Arguments: {args}")
    logger.info(f"Arguments: {args}")

    assert args.task in Tasks.values(), f"Task must be one of {Tasks.values()}"
    assert args.encoding_method in EncodingMethods.values(), f"Encoding method must be one of {EncodingMethods.values()}"

    # Set the device
    device = torch.device(args.device)

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, token=HF_TOKEN).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Remove the today's date from the prompt for reproducibility
    if args.model in ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]:
        tokenizer.chat_template = LLAMA_CHAT_TEMPLATE

    # Create steering config
    steering_config = SteeringConfig(tokenizer=tokenizer, **vars(args))
    # print(f"Steering configuration: {steering_config}")
    logger.info(f"Steering configuration: {steering_config}")

    # Load and preprocess the dataset
    dataset = load_task_dataset(args.task, args.subtask)

    # Construct the evaluation dataset
    prefix_to_add = args.prefix if args.append_prefix_to_prompt else None
    eval_constructor = EvaluationDatasetConstructor(
        dataset=dataset["test"],
        tokenizer=tokenizer,
        n=args.n,
        num_fewshot_prompt=args.num_fewshot_prompt,
        task=args.task,
        prefix=prefix_to_add,
        system_prompt=args.system_prompt,
        encoding_method=args.encoding_method,
        add_generation_prompt=args.add_generation_prompt,
    )
    evaluation_dataset = eval_constructor.construct_dataset()

    if args.verbose:
        # print(f"First example in the original evaluation subset: {dataset['test'][0]}")
        # print(f"First example in evaluation dataset after preprocessing: {evaluation_dataset[0]}")
        logger.info(f"First example in the original evaluation subset: {dataset['test'][0]}")
        logger.info(f"First example in evaluation dataset after preprocessing: {evaluation_dataset[0]}")

    # Define the generation arguments for experiments
    max_new_tokens = args.max_new_tokens
    if args.eval_type == "greedy":
        experiments = [{"do_sample": False, "max_new_tokens": max_new_tokens}]

    elif args.eval_type == "sampling":
        gen_kwargs = get_gen_kwargs(model)
        gen_kwargs["max_new_tokens"] = max_new_tokens
        experiments = [gen_kwargs]

    for temp in args.temperatures:
        sub_experiment = deepcopy(experiments[0])
        sub_experiment["temperature"] = temp
        sub_experiment["do_sample"] = True
        experiments.append(sub_experiment)
    del args.temperatures

    for generation_kwargs in experiments:

        # Create an experiment id to keep track of the runs
        experiment_id = str(uuid.uuid4())
        # print(f"Experiment ID: {experiment_id}")
        # print(f"Experiment generation arguments: {generation_kwargs}")
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info(f"Experiment generation arguments: {generation_kwargs}")

        # Run the evaluation multiple times
        for run_idx in range(args.n_runs):

            args.seed = args.seed + run_idx
            # print(f"Run {run_idx + 1}/{args.n_runs} with seed {args.seed}")
            logger.info(f"Run {run_idx + 1}/{args.n_runs} with seed {args.seed}")

            args_dict = vars(args).copy()
            args_dict.update(generation_kwargs)

            # Set seeds for reproducibility
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            random.seed(args.seed)
            steering_config.seed = args.seed

            # Construct the contrastive dataset
            contrastive_constructor = ContrastiveDatasetConstructor(
                dataset=dataset["train"], steering_config=steering_config, task=args.task
            )
            contrastive_dataset = contrastive_constructor.construct_dataset()
            if args.verbose:
                # print(f"First example in the contrastive dataset: {contrastive_dataset[0]}")
                logger.info(f"First example in the contrastive dataset: {contrastive_dataset[0]}")

            # Evaluate the model
            evaluator = Evaluator(
                model=model,
                tokenizer=tokenizer,
                dataset=evaluation_dataset,
                task=args.task,
                device=args.device,
                encoding_method=args.encoding_method,
                steering_config=steering_config,
                extraction_dataset=contrastive_dataset,
                force_choice=args.force_choice,
                cache_dir=args.vector_cache_dir,
                subtask=args.subtask,
            )

            results = evaluator.evaluate(
                batch_size=args.batch_size,
                generation_kwargs=deepcopy(generation_kwargs),
            )
            args_dict["vector_id"] = evaluator.get_vector_id()

            if isinstance(results, dict):
                logger.info(f"Results: {results['metrics']}")

            # Get logs from memory handler
            logs = log_stream.getvalue()

            if isinstance(results, list):
                for i, result in enumerate(results):
                    args_dict_copy = args_dict.copy()
                    args_dict_copy["cluster_id"] = i
                    # Add logs to metadata
                    result["metadata"] = {
                        "fewshot_map": contrastive_constructor.fewshot_map,
                        "logs": logs,
                    }
                    save_experiment_results(result, args_dict_copy, experiment_id, args.task, is_baseline=False)
            else:
                # Add logs to metadata
                results["metadata"] = {
                    "fewshot_map": contrastive_constructor.fewshot_map,
                    "logs": logs,
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                }
                save_experiment_results(results, args_dict, experiment_id, args.task, is_baseline=False)
