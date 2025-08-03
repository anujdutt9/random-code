from enum import Enum
from typing import Any

# Parameters that affect the creation steering vector
VECTOR_AFFECTING_PARAMETERS = [
    "add_question",
    "add_prefix",
    "add_answer",
    # "sample_fewshot_examples",
    "num_fewshot_examples",
    "n_contrastive_samples",
    "take_difference",
    "extraction_method",
    "prefix",
    "padding_side",
    "seed",
    "aggregation_method",
    "layers_ids_keys",
    "layers_ids_values",
    "fewshot_delimeter",
    "fewshot_only_in_postive",
    "use_tokenized_version",
    "encoding_method",
    "add_special_tokens",
    "extraction_system_prompt",
    "add_generation_prompt",
    "clustering_method",
    "cluster_layer_id",
    "cluster_on",
    "n_clusters",
    "sample_selection_method",
]

ANSWER_EXTRACTION_PROMPT = "So the correct choice is"


class ExtendedEnumMixin(Enum):
    @classmethod
    def keys(cls) -> list[str]:
        return [attr.name for attr in cls]

    @classmethod
    def values(cls) -> list:
        return [attr.value for attr in cls]

    @classmethod
    def items(cls) -> dict[str, Any]:
        return {attr.name: attr.value for attr in cls}
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value


class ExtractionMethods(str, ExtendedEnumMixin):
    last_token = "last_token"
    sequence_mean = "sequence_mean"
    prefix_mean = "prefix_mean"


class Tasks(str, ExtendedEnumMixin):
    gsm8k = "gsm8k"
    gsm8k_oai = "gsm8k-oai"
    csqa = "csqa"
    csqa_oai = "csqa-oai"
    arc_oai = "arc-oai"
    piqa_oai = "piqa-oai"


class EncodingMethods(str, ExtendedEnumMixin):
    qa = "qa"
    instruct = "instruct"


class AggregationMethods(str, ExtendedEnumMixin):
    clustering = "clustering"                   # k-means clustering of the steering vectors over all examples
    mean = "mean"                               # mean of the steering vectors over all examples
    none = "none"                               # no aggregation, returns a steering vector for each example


class SampleSelectionMethods(str, ExtendedEnumMixin):
    random = "random"                         # randomly select n examples from the dataset
    distance = "distance"                     # select n examples based on the distance to the question


LLAMA_CHAT_TEMPLATE = new_chat_template = """
{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""
