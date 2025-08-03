from abc import ABC, abstractmethod
import re

from src.utils.constants import Tasks, ANSWER_EXTRACTION_PROMPT
from src.utils.helpers import extract_choices, strip_chat_template


class Filter(ABC):
    def __init__(
        self,
        regex_patterns: list[str],
        name: str,
        group_select: int = -1,
        fallback_uncertainty_symbol: str = "[invalid]",
        fallback_incorrect_symbol: str = "[incorrect]",
    ):
        self.regex_patterns = [re.compile(p) for p in regex_patterns]
        self.group_select = group_select
        self.fallback_uncertainty_symbol = fallback_uncertainty_symbol
        self.fallback_incorrect_symbol = fallback_incorrect_symbol
        self.name = name

        self.incorrect_pattern = re.compile(
            r"\b("
            r"none|no correct|no answer|no option|none of the (?:above|choices|options)|"
            r"(?:options?|choices?)\s+[A-E](?:\s*(?:,|or|and)\s*[A-E])+|"
            r"[A-E](?:\s*(?:,|or|and)\s*[A-E])+|"
            r"either\s+[A-E]\s+or\s+[A-E]|"
            r"combination of"
            r")\b",
            re.IGNORECASE,
        )

    def __call__(self, text, question=None):
        text = self.preprocess(text)
        for regex in self.regex_patterns:
            match = regex.findall(text)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                return self.normalize_text(match.strip())

        # Select only the last part of the text if it contains the answer extraction prompt
        if ANSWER_EXTRACTION_PROMPT in text:
            text = (
                ANSWER_EXTRACTION_PROMPT + text.split(ANSWER_EXTRACTION_PROMPT)[-1]
            )

        # If no match is found with regex labels, try to match with choices
        if question:
            # Extract choices from the question
            choices = extract_choices(strip_chat_template(question))

            # Reject sentence if it includes negation
            if any(neg in text.lower() for neg in ["not", "n't", "incorrect", "wrong"]):
                return self.fallback_incorrect_symbol

            # Count matching choices
            matched = [
                letter
                for letter, phrase in choices.items()
                if phrase.lower() in text.lower()
            ]

            # Return only if exactly one match
            if len(matched) == 1:
                return self.normalize_text(matched[0])

        # Explicitly detect multiple/none answers if the answer extraction prompt is present
        if ANSWER_EXTRACTION_PROMPT in text and self.incorrect_pattern.search(text):
            return self.fallback_incorrect_symbol

        return self.fallback_uncertainty_symbol

    @abstractmethod
    def normalize_text(self, text):
        pass

    @abstractmethod
    def preprocess(self, text):
        pass


class GSM8KFilter(Filter):
    def normalize_text(self, text: str):
        text = text.replace(",", "")
        text = text.replace(".", "")
        text = text.replace("$", "")
        return text

    def preprocess(self, text: str):
        return text


class CSQAFilter(Filter):
    def normalize_text(self, text: str):
        return text

    def preprocess(self, text: str):
        text = text.replace("\n", " ")
        # text = text.replace(":", " ")
        text = text.replace("-", " ")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("**", " ")

        # Remove double spaces
        text = re.sub(r"\s+", " ", text)
        return text


class ArcFilter(Filter):
    def normalize_text(self, text: str):
        return text

    def preprocess(self, text: str):
        text = text.replace("\n", " ")
        # text = text.replace(":", " ")
        text = text.replace("-", " ")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("**", " ")

        # Remove double spaces
        text = re.sub(r"\s+", " ", text)
        return text
    

class PIQAFilter(Filter):
    def normalize_text(self, text: str):
        return text

    def preprocess(self, text: str):
        text = text.replace("\n", " ")
        # text = text.replace(":", " ")
        text = text.replace("-", " ")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("**", " ")

        # Remove double spaces
        text = re.sub(r"\s+", " ", text)
        return text


PATTERNS = {
    "multiple_choice": [
        r"[Tt]he correct choice is:?\s*([A-E])\b",
        r"[Tt]he correct answer is:?\s*([A-E])\b",
        r"\b[Tt]he correct answer is:?\s*(?:(?!not|n't|incorrect|wrong|and|or|combination|all of|[,;]).){0,50}?\b([A-E])\b(?![^A-E]*\b[A-E]\b)",
        r"\b[Tt]he best answer is:?\s*([A-E])(?:\b|(?=[\.\,\;\:\s]|$))",
        r"[Tt]he correct choice is option:?\s*([A-E])(?:\b|(?=[\.\,\;\:\s]|$))",
        r"\b[Tt]he final answer is:?\s*([A-E])(?:\b|(?=[\.\,\;\:\s]|$))",
        r"[Tt]he answer is:?\s*([A-E])\b",
        r"[Ff]inal answer is:?\s*([A-E])(?:\b|(?=[\.\,\;\:\s]|$))",
        r"[Tt]he best answer is option:?\s*([A-E])(?:\b|(?=[\.\,\;\:\s]|$))",
        r"[Tt]he correct choice is:?\s*(?:\n\s*)*Choice\s+([A-E])\s*:",
        r"[Aa]nswer:\s*([A-E])\b",
        r"^([A-E])\s*:",
    ],
    "digit": [r"(-?[$0-9.,]{2,})|(-?[0-9]+)"],
}

FILTERS = {
    Tasks.gsm8k: [
        GSM8KFilter(PATTERNS["digit"], name="last_digit"),
        GSM8KFilter([r"The answer is (\-?[0-9\.\,]+)."], name="strict_match"),
    ],
    Tasks.gsm8k_oai: [
        GSM8KFilter(PATTERNS["digit"], name="last_digit"),
        GSM8KFilter([r"The answer is (\-?[0-9\.\,]+)."], name="strict_match"),
    ],
    Tasks.csqa: [
        CSQAFilter(PATTERNS["multiple_choice"], name="augmented_extract"),
        CSQAFilter(["([A-E])"], name="last_letter"),
    ],
    Tasks.csqa_oai: [
        CSQAFilter(PATTERNS["multiple_choice"], name="augmented_extract"),
        CSQAFilter(["([A-E])"], name="last_letter"),
    ],
    Tasks.arc_oai: [
        ArcFilter(PATTERNS["multiple_choice"], name="augmented_extract"),
        ArcFilter(["([A-E])"], name="last_letter"),
    ],
    Tasks.piqa_oai: [
        PIQAFilter(PATTERNS["multiple_choice"], name="augmented_extract"),
    ],
}
