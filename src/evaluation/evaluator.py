"""
Core evaluation functionality for LLM models.
"""

import os
from tqdm import tqdm
from copy import deepcopy

import torch

from src.evaluation.metrics import compute_metrics
from src.utils.constants import (
    EncodingMethods,
    Tasks,
    AggregationMethods,
    ANSWER_EXTRACTION_PROMPT,
)
from src.utils.helpers import (
    select_steering_kv_layers,
    generate_vector_id,
    load_vector,
    save_vector,
    compute_example_length,
    ForceChoiceProcessor,
)
from src.steering.cache_steering import (
    extract_steering_kv,
    generate_with_cache_steering,
)
from src.utils.logging_setup import logger

MC_TASKS = [
    Tasks.csqa,
    Tasks.csqa_oai,
    Tasks.arc_oai,
    Tasks.piqa_oai,
]


class Evaluator:
    """
    Evaluator class for running experiments on LLM models.
    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        task,
        device,
        encoding_method="instruct",
        steering_config=None,
        extraction_dataset=None,
        cache_dir="cached_vectors",
        force_choice=False,
        subtask=None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            task: Task to evaluate on
            device: Device to run the evaluation on
            encoding_method: Encoding method for the dataset
            steering_config: Steering configuration, if steering is enabled, otherwise None
            extraction_dataset: Dataset to use for steering vector extraction
            subtask: Subtask to evaluate on, if applicable
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.task = task
        self.subtask = subtask
        self.device = device
        self.encoding_method = encoding_method
        self.cache_dir = cache_dir
        self.stop_strings = ["Q:", "Question:", "</s>", "<|im_end|>"]
        
        self.force_choice = force_choice
        allowed_letters = ["A", " A", "B", " B", "C", " C", "D", " D", " E", "E"]
        self.allowed_token_ids = [tokenizer(l, add_special_tokens=False)['input_ids'][0] for l in allowed_letters]

        # If steering config is passed, cache steering is enabled, otherwise it is disabled
        self.steering_config = steering_config
        self.extract_steering_vectors(extraction_dataset)

        if self.tokenizer.eos_token not in self.stop_strings:
            self.stop_strings.append(self.tokenizer.eos_token)

        # if torch.cuda.device_count() > 1:
        #     # Initialize Accelerator
        #     accelerator = Accelerator()
        #     logger.info(f"Device(s) used by accelerator: {accelerator.device}")
        #     self.model = accelerator.prepare(self.model)
        #     self.device = accelerator.device

    def tokenize(self, text):
        """
        Tokenize text using the model's tokenizer.
        
        Args:
            text (str): Text to tokenize
        
        Returns:
            torch.Tensor: Tokenized text
        """
        if self.encoding_method == EncodingMethods.qa:
            add_special_tokens = True
        elif self.encoding_method == EncodingMethods.instruct:
            add_special_tokens = False

        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=add_special_tokens,
        )

    def generate(self, tokens, **kwargs):
        """
        Generate a response from the model.
        
        Args:
            prompt (str): Input prompt
            temperature (float): Temperature for sampling
            
        Returns:
            str: Model response
        """

        if self.steering_config:
            response = generate_with_cache_steering(
                self.model,
                tokens,
                self.steering_kv,
                self.steering_config,
                False,
                task=self.task,
                **kwargs
            )
        else:
            response = self.model.generate(tokens, **kwargs)

        if self.task in MC_TASKS:
            response = self.generate_answer_span(response, kwargs.get("attention_mask"))

        return response

    def generate_answer_span(self, response, attention_mask):
        tokens_to_append = self.tokenizer(
            f"\n{ANSWER_EXTRACTION_PROMPT}", return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        append_input_ids = tokens_to_append["input_ids"]

        # Create new responses with EOS token removed and ANSWER_EXTRACTION_PROMPT appended
        modified_responses = []
        for i, r in enumerate(response):

            attn_mask = attention_mask[i]
            n_generated_tokens = r.size(0) - attn_mask.size(0)

            # Remove all EoS tokens at the end if present
            n_tokens_to_remove = 0
            end_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
            if self.tokenizer.name_or_path in [
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
            ]:
                end_token_ids.append(128001)

            for token_idx in range(-1, -n_generated_tokens, -1):
                if r[token_idx] in end_token_ids:
                    n_tokens_to_remove += 1
                else:
                    # Stop at first non-end token
                    break
            if n_tokens_to_remove > 0:
                r = r[:-n_tokens_to_remove]

            # Append the ANSWER_EXTRACTION_PROMPT tokens
            new_input_ids = torch.cat([r, append_input_ids[0]], dim=0).unsqueeze(0)
            old_attn_mask = attn_mask.unsqueeze(0)
            n_added_tokens = r.size(0) + append_input_ids.size(1) - old_attn_mask.size(1)
            new_attn_mask = torch.cat([old_attn_mask, torch.ones(1, n_added_tokens, device=self.device)], dim=-1)

            # Set the logits processor to force the model to choose from the allowed token ids if force_choice is enabled
            logits_processor = None
            max_new_tokens = 10
            # if self.force_choice:
            #     logits_processor = [ForceChoiceProcessor(self.allowed_token_ids)]
            #     max_new_tokens = 1

            # Generate the additional response
            additional_response = self.model.generate(
                new_input_ids,
                attention_mask=new_attn_mask,
                use_cache=True,
                stop_strings=self.stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
            )[0]

            modified_responses.append(additional_response)
        return modified_responses

    def preprocess_dataset(self, batch_size):
        """Pre-tokenize all examples in the dataset for faster evaluation"""

        # Sort the dataset by length of input for faster evaluation
        self.dataset = self.dataset.map(compute_example_length)
        self.dataset = self.dataset.sort("length", reverse=True, keep_in_memory=True)

        processed_examples = []
        for example in tqdm(self.dataset.iter(batch_size=batch_size), desc="Pre-tokenizing"):
            tokenized = self.tokenize(example["input"])
            processed_examples.append({
                "id": example["id"],
                "tokens": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "raw_input": example["input"],
                "answer": example["answer"]
            })

        return processed_examples

    def evaluate(self, batch_size=1, generation_kwargs=None, use_cache=True):
        if (
            self.steering_config
            and self.steering_config.aggregation_method == AggregationMethods.clustering
        ):
            return self._evaluate_with_clustering(
                batch_size=batch_size,
                generation_kwargs=generation_kwargs,
                use_cache=use_cache,
            )
        else:
            if self.steering_config:
                self.steering_kv = select_steering_kv_layers(self.steering_kv, self.steering_config)
            return self._evaluate(
                batch_size=batch_size,
                generation_kwargs=generation_kwargs,
                use_cache=use_cache,
            )

    def _evaluate(self, batch_size=1, generation_kwargs=None, use_cache=True):
        """
        Evaluate the model on a dataset.

        Args:
            batch_size (int): Batch size for evaluation
            generation_kwargs (dict): Additional keyword arguments for generation
            
        Returns:
            dict: Results of the evaluation
        """
        generation_kwargs = generation_kwargs or {}
        all_responses = []

        # Pre-tokenize everything once
        processed_examples = self.preprocess_dataset(batch_size=batch_size)

        for example in tqdm(processed_examples, desc="Generating responses"):

            # Generate response
            responses = self.generate(
                example["tokens"].to(self.device),
                attention_mask=example["attention_mask"].to(self.device),
                use_cache=use_cache,
                stop_strings=self.stop_strings,
                tokenizer=self.tokenizer,
                **generation_kwargs
            )
            example['response'] = responses

        # Decode the responses and save the inputs adn targets
        for example in processed_examples:
            for i in range(len(example["raw_input"])):
                input_tokens_len = example["tokens"][i].size(0)
                generated_tokens = example["response"][i][input_tokens_len:]
                response = self.tokenizer.decode(generated_tokens)

                all_responses.append(
                    {   
                        "id": example["id"][i],
                        "input": example["raw_input"][i],
                        "target": example["answer"][i],
                        "response": response,
                    }
                )

        metrics = compute_metrics(all_responses, task=self.task)

        if self.force_choice and self.task in MC_TASKS:
            logger.info("Forcing choice in the responses")
            logger.info(f"Old metrics: {metrics}")
            all_responses, metrics = self._force_choice(all_responses)
            logger.info(f"New metrics: {metrics}")

        results = {
            "metrics": metrics,
            "samples": all_responses,
        }

        return results

    def _evaluate_with_clustering(self, batch_size=1, generation_kwargs=None, use_cache=True):
        steering_kv_copy = deepcopy(self.steering_kv)
        cluster_results = []

        unique_labels, counts = torch.unique(
            steering_kv_copy["labels"], return_counts=True
        )
        label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
        logger.info(f"Label counts: {label_counts}")

        for i in range(self.steering_config.n_clusters):
            logger.info(f"Evaluating cluster {i + 1}/{self.steering_config.n_clusters}")
            self.steering_kv["values"] = steering_kv_copy["values"][i]
            self.steering_kv["keys"] = steering_kv_copy["keys"][i]
            self.steering_kv = select_steering_kv_layers(self.steering_kv, self.steering_config)

            results = self._evaluate(batch_size=batch_size, generation_kwargs=generation_kwargs, use_cache=use_cache)
            cluster_results.append(results)

        self.steering_kv = steering_kv_copy
        return cluster_results
    

    def _force_choice(self, all_responses):
        """
        Force the model to choose from the allowed token ids.

        Args:
            all_responses (list): List of all responses to process
        """
        for example in tqdm(all_responses, desc="Forcing choice"):
            if example["filtered_response"]["augmented_extract"] != "[invalid]":
                continue
            logger.info(f"Forcing choice in example: {example['id']}")
            new_input = example["input"] + example["response"].split(ANSWER_EXTRACTION_PROMPT)[0] + ANSWER_EXTRACTION_PROMPT
            tokens = self.tokenize(new_input)
            logits_processor = [ForceChoiceProcessor(self.allowed_token_ids)]

            response = self.model.generate(
                tokens["input_ids"].to(self.device),
                attention_mask=tokens["attention_mask"].to(self.device),
                use_cache=True,
                stop_strings=self.stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False,
                max_new_tokens=1,
                logits_processor=logits_processor,
            )
            input_tokens_len = self.tokenize(example["input"])['input_ids'].shape[1]
            generated_tokens = response[0][input_tokens_len:]
            example["response"] = self.tokenizer.decode(generated_tokens)
        metrics = compute_metrics(all_responses, task=self.task)
        return all_responses, metrics


    def extract_steering_vectors(self, extraction_dataset):
        """
        Extract steering vectors from the dataset if steering is enabled.

        Args:
            extraction_dataset: Dataset to use for steering vector extraction
        """
        if not self.steering_config:
            self.steering_kv = None
            return

        # Generate vector id
        self.vector_id = generate_vector_id(self.steering_config, self.tokenizer.name_or_path, self.task, self.subtask)
        if os.path.exists(f"{self.cache_dir}/{self.vector_id}.pt"):
            logger.info(f"Loading steering vector from cache: {self.vector_id}")
            self.steering_kv = load_vector(self.vector_id, self.cache_dir, device=self.device)
            return

        logger.info(f"Generating steering vector: {self.vector_id}")
        self.steering_kv = extract_steering_kv(
            self.model,
            self.tokenizer,
            extraction_dataset,
            self.steering_config,
            batch_size=1,
            device=self.device,
        )
        save_vector(self.steering_kv, self.vector_id, self.cache_dir)

    def get_vector_id(self):
        """
        Get the vector ID of the steering vector.
        
        Returns:
            str: Vector ID
        """
        return self.vector_id
