#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import re
from typing import List, Dict, Any, Optional, Iterable, Tuple

from weak_annotators.model_registry import ModelsRegistry
from transformers import LlamaTokenizer
from vllm import LLM, SamplingParams, LLMEngine
import jinja2

from weak_annotators.base_extractor import BaseExtractor, Span

logger = logging.getLogger(__name__)


# monkey patching to avoid GPU limitation ################################
def _init_cache(self) -> None:
    """Profiles the memory usage and initializes the KV cache."""
    # Get the maximum number of blocks that can be allocated on GPU and CPU.
    num_blocks = self._run_workers(
        "profile_num_available_blocks",
        block_size=self.cache_config.block_size,
        gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
        cpu_swap_space=self.cache_config.swap_space_bytes,
    )

    # Since we use a shared centralized controller, we take the minimum
    # number of blocks across all workers to make sure all the memory
    # operators can be applied to all workers.
    num_gpu_blocks = min(b[0] for b in num_blocks)
    num_cpu_blocks = min(b[1] for b in num_blocks)
    logger.info(f"# GPU blocks: {num_gpu_blocks}, " f"# CPU blocks: {num_cpu_blocks}")

    if num_gpu_blocks <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when "
            "initializing the engine."
        )
    max_seq_len = self.cache_config.block_size * num_gpu_blocks
    self.model_config.max_model_len = 1024  # hack to avoid GPU limitation for now

    if self.model_config.max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({self.model_config.max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine."
        )

    self.cache_config.num_gpu_blocks = num_gpu_blocks
    self.cache_config.num_cpu_blocks = num_cpu_blocks

    # Initialize the cache.
    self._run_workers("init_cache_engine", cache_config=self.cache_config)
    # Warm up the model. This includes capturing the model into CUDA graph
    # if enforce_eager is False.
    self._run_workers("warm_up_model")


LLMEngine._init_cache = _init_cache

######################################################################

KEY_VALUE_SPLITTER_REGEX = re.compile(r"\s*[-:]\s*", re.IGNORECASE)


def key_value_splitter(text: str) -> Iterable[Tuple[str, str, str]]:
    for match in KEY_VALUE_SPLITTER_REGEX.finditer(text):
        yield text[:match.start()].strip(), match.group().strip(), text[match.end():].strip()


class MedAlpacaExtractor(BaseExtractor):
    model_name = "medalpaca/medalpaca-7B"
    max_new_tokens: int = 256
    tensor_parallel_size: int = 1
    max_input_length: int = 512

    def __init__(
            self,
            labels: List[str],
            labels_descriptions: Dict[str, str],
            examples: Optional[Dict[str, List[str]]] = None,
            prompt_template: Optional[str] = None, **kwargs
    ):
        """

        :param labels: list of labels (e.g. ["PER", "LOC", "ORG"])
        :param labels_descriptions:  optional dictionary with labels descriptions (e.g. {"PER": "Person", "LOC": "Location", "ORG": "Organization"})
        :param examples: optional dictionary with examples (e.g. {"London is a capital of Great Britain": "London-LOC\nGreat Britain-LOC"})
        :param prompt_template: optional prompt template
        :param kwargs:
        """
        super().__init__(labels=labels, labels_descriptions=labels_descriptions, **kwargs)
        if not ModelsRegistry.has(self.model_name):
            llm = LLM(
                model=self.model_name, tensor_parallel_size=self.tensor_parallel_size
            )
            ModelsRegistry.register_model(self.model_name, llm)

        self.llm = ModelsRegistry.get_model(self.model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.prompt_template = prompt_template
        if self.prompt_template is None:
            self.prompt_template = self._build_prompt(examples)
        self.prompt_template = self.prompt_template.strip()

    def get_response(self, responses):
        responses = [r.split("ASSISTANT:")[-1].strip() for r in responses]
        return responses

    def inference(
            self,
            prompts: List[str],
            max_new_tokens: int = 256,
    ):
        sampling_params = SamplingParams(
            temperature=0, max_tokens=max_new_tokens, stop=["</s>"]
        )
        responses = self.llm.generate(prompts, sampling_params)
        responses_corret_order = []
        response_set = {response.prompt: response for response in responses}
        for prompt in prompts:
            if prompt not in response_set:
                logger.info(f"Prompt not found in response: {prompt}")
                continue
            responses_corret_order.append(response_set[prompt])
        responses = responses_corret_order
        outputs = self.get_response([output.outputs[0].text for output in responses])
        return outputs

    def _extract(self, text: str, **kwargs: Any) -> List[Span]:
        samples = [self.prompt_template.format(text.replace("\n", " "))]

        outputs = self.inference(samples, max_new_tokens=self.max_new_tokens)[0]

        if outputs.startswith("Entities:"):
            outputs = outputs.replace("Entities:", "").strip()

        spans = []
        for output in outputs.split("\n"):
            for span_label, _, span_text in key_value_splitter(output):
                if not span_text or not span_label:
                    continue

                if span_text not in text and span_label in text:
                    span_text, span_label = span_label, span_text

                if not span_text in text:
                    continue

                spans.append(
                    Span(
                        start=text.index(span_text),
                        end=text.index(span_text) + len(span_text),
                        text=span_text,
                        label=span_label,
                    )
                )
        return spans

    def _build_prompt(self, examples: Dict[str, List[str]]) -> str:
        template = """
        I need your expertise in Named Entity Recognition (NER) for biomedical domain with a 
focus on accurately identifying entities and their acronyms. Each identified entity should be 
classified into one of the following categories: Treatment, TableID, ProtocolID, or Disease. 
For acronyms, please indicate the full term alongside the acronym and classify both consistently. 
Here are the categories defined:
{% for label in labels %}
{{label}}: {{labels_descriptions[label]}}.
{% endfor %}
{% if examples %}
Example
{% for text, entities in examples.items() %}
Text: {{text}}
Entities: {{entities}}
{% endfor %}
{% endif %}
Now, based on the structure and classification shown in the examples above, please analyze the following text and identify the entities with their correct classification if any. 
Use comma-separated values (CSV with - separator) format to report your findings and do not repeat the same text entity.
Text: {}
Entities:""".strip()
        template = jinja2.Template(template)
        prompt_template = template.render(
            labels=self.labels, labels_descriptions=self.labels_descriptions, examples=examples
        )
        prompt_template = "\n".join([line for line in prompt_template.split("\n") if line.strip()])
        return prompt_template


if __name__ == "__main__":
    labels = ["protocol_id", "treatment", "table_id", "disease"]
    labels_descriptions = {
        "protocol_id": "protocol id or number",
        "treatment": "Treatment or drug",
        "table_id": "Table ID or number",
        "disease": "Disease or condition",
    }
    ner_pipeline = MedAlpacaExtractor(labels, labels_descriptions=labels_descriptions)
    text = """
        Table 14.2.1.6.1
                Tanezumab Protocol A4091056
                Summary of Analysis of Change from Baseline for WOMAC Pain Subscale by MMRM by Week (ITT, Observed Data)
        """.strip()

    print(ner_pipeline(text))
