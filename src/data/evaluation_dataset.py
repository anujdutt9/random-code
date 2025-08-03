import logging

from datasets import Dataset

from src.data.dataset import DatasetConstructor

logger = logging.getLogger(__name__)


class EvaluationDatasetConstructor(DatasetConstructor):

    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        n: int,
        num_fewshot_prompt: int,
        task: str,
        prefix: str = None,
        encoding_method="instruct",
        use_tokenized_version=False,
        system_prompt=None,
        add_generation_prompt=False,
    ):
        super().__init__(
            dataset,
            task,
            tokenizer,
            use_tokenized_version,
            encoding_method,
        )
        self.num_fewshot = num_fewshot_prompt
        self.prefix = prefix
        self.system_prompt = system_prompt
        self.add_generation_prompt = add_generation_prompt

        if n:
            self.dataset = self.dataset.select(range(n))

    def sample_fewshot_examples(self, example):
        return self.fewshot_examples[:self.num_fewshot]

    def process_qa_example(self, example, fewshot_examples):
        input = ""

    def process_instruct_example(self, example, fewshot_examples):
        input = []

        # Add the system prompt
        if self.system_prompt and not "raise_exception('System role not supported')" in self.tokenizer.chat_template: # TODO: there has to be a better solution than hardcoding this
            input.append({"role": "system", "content": self.system_prompt})

        # Add the few-shot examples
        for fewshot_example in fewshot_examples:
            question = self.question_template.format(question=fewshot_example["question"])
            input.append({"role": "user", "content": question})

            answer = fewshot_example["steps"] + self.answer_template.format(answer=fewshot_example["answer"])
            input.append({"role": "assistant", "content": answer})

        # Add the actual example
        input.append({"role": "user", "content": self.question_template.format(question=example["question"])})

        # Apply chat template
        input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=self.add_generation_prompt)

        # Add prefix if needed
        if self.prefix:
            input += self.prefix

        return input
