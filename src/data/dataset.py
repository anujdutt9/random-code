from abc import ABC, abstractmethod
import yaml

from datasets import Dataset

from src.utils.constants import EncodingMethods, Tasks


class DatasetConstructor(ABC):
    """Abstract base class for dataset handling operations."""

    def __init__(
        self,
        dataset: Dataset,
        task: str,
        tokenizer,
        use_tokenized_version: bool = False,
        encoding_method: str = "qa",
    ):
        """Initialize the Dataset class."""
        self.dataset = dataset
        self.task = task
        self.encoding_method = encoding_method
        self.tokenizer = tokenizer
        self.fewshot_examples = []

        # Set the template
        if self.encoding_method == EncodingMethods.qa:
            if use_tokenized_version:
                self.template = ["Question:", " ", "{question}", "\n", "Answer:" , "{answer}"]
            else:
                self.template = "Question: {question}\nAnswer:"
                self.answer_template = " The answer is {answer}."

        elif self.encoding_method == EncodingMethods.instruct:
            self.question_template = "{question}"
            self.answer_template = " The answer is {answer}."

    def construct_dataset(self):
        return self.dataset.map(self.process_example, keep_in_memory=True)

    def process_example(self, example):

        # Select fewshot examples
        fewshot_examples = self.sample_fewshot_examples(example)

        # Process the example
        if self.encoding_method == EncodingMethods.qa:
            processed_examples = self.process_qa_example(example, fewshot_examples)
        elif self.encoding_method == EncodingMethods.instruct:
            processed_examples = self.process_instruct_example(example, fewshot_examples)

        if isinstance(processed_examples, tuple):
            return {"positive": processed_examples[0], "negative": processed_examples[1]}
        else:
            return {"input": processed_examples}

    @abstractmethod
    def process_qa_example(self, example, fewshot_examples):
        pass

    @abstractmethod
    def process_instruct_example(self, example, fewshot_examples):
        pass

    @abstractmethod
    def sample_fewshot_examples(self, example):
        pass
