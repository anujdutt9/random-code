from abc import ABC, abstractmethod
from typing import Dict, Any
import yaml
import logging
import re
import uuid

from datasets import load_dataset, Dataset

from src.utils.constants import Tasks

logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """Abstract base class for dataset handling operations."""

    def __init__(self):
        """Initialize the Dataset class."""
        self._load_config()

    @abstractmethod
    def load_dataset(self) -> None:
        """Load the dataset into memory."""
        pass

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.CONFIG_PATH, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        self.fewshot_samples = self.config.get("fewshot_samples")


class GSM8KLoader(DatasetLoader):
    """
    Dataset class for handling GSM8K math problem datasets.
    Dataset format: {"question": str, "steps": List[str], "answer": str}
    """

    def __init__(self, task: str = "gsm8k"):
        """
        Initialize the GSM8K dataset.

        Args:
            task: Which task variant to load (default: "gsm8k")
        """
        self.task = task

    def _format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a GSM8K example for the QA encoding method.

        Args:
            example: The example to format
        Returns:
            Formatted example
        """
        cot_chain, numerical_answer = example["answer"].split("#### ")
        numerical_answer = numerical_answer.replace(",", "")
        cot_chain = cot_chain.replace("\n", " ")
        cot_chain = re.sub(r"<<.*?>>", "", cot_chain)
        cot_chain = cot_chain.strip()
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, example["question"])),
            "question": example["question"],
            "steps": cot_chain,
            "answer": numerical_answer,
        }

    def _format_oai_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a GSM8K example to match the dataset format.

        Args:
            example: The example to format
        Returns:
            Formatted example
        """
        cot_chain = " ".join(example["steps"])
        numerical_answer = example["solution"].replace(",", "")
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, example["query"])),
            "question": example["query"],
            "steps": cot_chain,
            "answer": numerical_answer,
        }

    def load_dataset(self) -> Dataset:
        """Load the GSM8K dataset and process it according to task type."""
        logger.info("Loading GSM8K dataset")

        dataset = load_dataset("gsm8k", "main")
        dataset = dataset.map(self._format_example, keep_in_memory=True)

        if self.task == Tasks.gsm8k_oai:
            train_set = load_dataset("json", data_files="local_data/gsm8k_oai/*.json")
            train_set = train_set.map(
                self._format_oai_example,
                remove_columns=["query", "solution"],
                keep_in_memory=True,
            )
            dataset["train"] = train_set["train"]
        return dataset


class CSQALoader(DatasetLoader):
    def __init__(self, task: str = "csqa"):
        """
        Initialize the Commonsense QA dataset.

        Args:
            task: Which task variant to load (default: "csqa")
        """
        self.task = task

    def load_dataset(self):
        """Load the Commonsense QA dataset."""
        logger.info("Loading CSQA dataset")

        dataset = load_dataset("tau/commonsense_qa", trust_remote_code=True)
        dataset = dataset.map(
            self._format_example,
            remove_columns=[
                "question_concept",
                "choices",
                "answerKey",
                # "reasoning_steps",  # Does not exist in the original dataset
                # "n_steps",          # Does not exist in the original dataset
            ],
            keep_in_memory=True,
        )

        if self.task == Tasks.csqa_oai:
            train_set = load_dataset("json", data_files="local_data/csqa_oai/*.json")
            train_set = train_set.map(
                self._format_oai_example,
                remove_columns=["query", "solution"],
                keep_in_memory=True,
            )
            dataset["train"] = train_set["train"]

        dataset["test"] = dataset["validation"]
        del dataset["validation"]

        return dataset

    def _format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a CSQA example to match the dataset format.

        Args:
            example: The example to format
        Returns:
            Formatted example
        """
        choices = "\n".join(
            [
                f"{l}: {t}"
                for l, t in zip(example["choices"]["label"], example["choices"]["text"])
            ]
        )
        question = example["question"] + "\n" + choices

        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, question)),
            "question": question,
            # "steps": example["reasoning_steps"],
            "steps": "",  # No reasoning steps in the original dataset
            "answer": example["answerKey"],
        }

    def _format_oai_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a CSQA example to match the dataset format.

        Args:
            example: The example to format
        Returns:
            Formatted example
        """
        question, choices = example["query"].split("Choices:")
        question = question.split("Question: ")[1]
        question = question.strip() + "\n" + choices.strip()
        cot_chain = " ".join(example["steps"])

        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, question)),
            "question": question,
            "steps": cot_chain,
            "answer": example["solution"],
        }


class ARCLoader(DatasetLoader):
    def __init__(self, task: str = "arc-oai", subtask: str = None):
        """
        Initialize the ARC-c dataset.

        Args:
            task: Which task variant to load (default: "arc-oai")
        """
        self.task = task
        self.dir_name = f"arc_{subtask}" if subtask is not None else "arc_oai"

    def load_dataset(self):
        """Load the ARC-c dataset."""
        logger.info("Loading ARC-c dataset")

        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        dataset = dataset.map(
            self._format_example,
            remove_columns=["choices", "answerKey"],
            keep_in_memory=True,
        )

        if self.task == Tasks.arc_oai:
            train_set = load_dataset("json", data_files=f"local_data/{self.dir_name}/*.json")
            train_set = train_set.map(
                self._format_oai_example,
                remove_columns=["query", "solution"],
                keep_in_memory=True,
            )
            dataset["train"] = train_set["train"]

        del dataset["validation"]

        return dataset

    def _format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a ARC-c example to match the dataset format.

        Args:
            example: The example to format
        Returns:
            Formatted example
        """
        choices = "\n".join(
            [
                f"{l}: {t}"
                for l, t in zip(example["choices"]["label"], example["choices"]["text"])
            ]
        )
        question = example["question"] + "\n\nChoices:\n" + choices

        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, question)),
            "question": question,
            "steps": "",
            "answer": example["answerKey"],
        }

    def _format_oai_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a CSQA example to match the dataset format.

        Args:
            example: The example to format
        Returns:
            Formatted example
        """
        question, choices = example["query"].split("Choices:")
        question = question.split("Question: ")[1]
        question = question.strip() + "\n\nChoices:" + "\n" + choices.strip()
        cot_chain = " ".join(example["steps"])

        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, question)),
            "question": question,
            "steps": cot_chain,
            "answer": example["solution"],
        }
    

class PIQALoader(DatasetLoader):
    def __init__(self, task: str = "piqa-oai"):
        """
        Initialize the PIQA dataset.

        Args:
            task: Which task variant to load (default: "piqa-oai")
        """
        self.task = task

    def load_dataset(self):
        """Load the PIQA dataset."""
        logger.info("Loading PIQA dataset")

        # dataset = load_dataset("ybisk/piqa", trust_remote_code=True)
        dataset = load_dataset("baber/piqa", trust_remote_code=True)

        # Set the validation set as the test set since the test set doesn't have labels
        dataset["test"] = dataset["validation"]
        del dataset["validation"]

        dataset = dataset.map(
            self._format_example,
            keep_in_memory=True,
        )

        if self.task == Tasks.piqa_oai:
            train_set = load_dataset("json", data_files="local_data/piqa_oai/*.json")
            train_set = train_set.map(
                self._format_oai_example,
                remove_columns=["solution"],
                keep_in_memory=True,
            )
            dataset["train"] = train_set["train"]

        return dataset

    def _format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a PIQA example to match the dataset format.

        Args:
            example: The example to format
        Returns:
            Formatted example
        """
        labels = ["A", "B"]
        choice_list = [
            example['sol1'],
            example['sol2']
        ]
        correct_index = example['label']
        choices = "\n".join([f"{l}: {t}" for l, t in zip(labels, choice_list)])
        question = example["goal"] + "\n\nChoices:\n" + choices

        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, question)),
            "question": question,
            "steps": "",
            "answer": labels[correct_index],
        }

    def _format_oai_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a PIQA example to match the dataset format.

        Args:
            example: The example to format
        Returns:
            Formatted example
        """
        question = example["question"]
        cot_chain = " ".join(example["steps"])

        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, question)),
            "question": question,
            "steps": cot_chain,
            "answer": example["solution"],
        }


def load_task_dataset(task: str, subtask: str = None) -> Dataset:
    """
    Load a dataset based on the task.

    Args:
        task: The task to load
    Returns:
        The loaded dataset
    """
    if task in [Tasks.gsm8k, Tasks.gsm8k_oai]:
        return GSM8KLoader(task).load_dataset()
    elif task in [Tasks.csqa, Tasks.csqa_oai]:
        return CSQALoader(task).load_dataset()
    elif task in [Tasks.arc_oai]:
        return ARCLoader(task, subtask).load_dataset()
    elif task in [Tasks.piqa_oai]:
        return PIQALoader(task).load_dataset()
    else:
        raise ValueError(f"Task {task} not supported")
