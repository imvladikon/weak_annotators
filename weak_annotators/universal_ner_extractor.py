#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from ast import literal_eval
from typing import List, Type, Dict, Optional, Any, Sequence

from transformers import LlamaTokenizer
from vllm import LLM, SamplingParams, LLMEngine

from weak_annotators.base_extractor import BaseExtractor, Span
from weak_annotators.model_registry import ModelsRegistry
from weak_annotators.templates.conversation import get_conv_template

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


def preprocess_instance(source):
    conv = get_conv_template("ie_as_qa")
    for j, sentence in enumerate(source):
        value = sentence["value"]
        if j == len(source) - 1:
            value = None
        conv.append_message(conv.roles[j % 2], value)
    prompt = conv.get_prompt()
    return prompt


def get_response(responses):
    responses = [r.split("ASSISTANT:")[-1].strip() for r in responses]
    return responses


class UniversalNerExtractor(BaseExtractor):
    model_name: str = "Universal-NER/UniNER-7B-type"
    max_new_tokens: int = 256
    tensor_parallel_size: int = 1
    max_input_length: int = 512

    def __init__(
            self,
            labels: List[str],
            labels_descriptions: Optional[Dict[str, str]] = None,
            prompt_template: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(
            labels=labels, labels_descriptions=labels_descriptions, prompt_template=prompt_template, **kwargs
        )

        if not ModelsRegistry.has(self.model_name):
            llm = LLM(
                model=self.model_name, tensor_parallel_size=self.tensor_parallel_size
            )
            ModelsRegistry.register_model(self.model_name, llm)

        self.llm = ModelsRegistry.get_model(self.model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.labels_descriptions = labels_descriptions or {}

    def inference(self, examples: Sequence[Dict]) -> Sequence[str]:
        prompts = [
            preprocess_instance(example["conversations"]) for example in examples
        ]
        sampling_params = SamplingParams(
            temperature=0, max_tokens=self.max_new_tokens, stop=["</s>"]
        )
        responses = self.llm.generate(prompts, sampling_params)
        responses_corret_order = []
        response_set = {response.prompt: response for response in responses}
        for prompt in prompts:
            assert prompt in response_set
            responses_corret_order.append(response_set[prompt])
        responses = responses_corret_order
        outputs = get_response([output.outputs[0].text for output in responses])
        return outputs

    def _one_shot(self, text: str, label: str) -> Sequence[Span]:
        # if len(self.tokenizer(text + entity_label)['input_ids']) > self.max_input_length:
        #     raise ValueError(
        #         f"Error: Input is too long. Maximum number of tokens for input and entity type is {self.max_input_length} tokens.")

        description = (
            f"({self.labels_descriptions[label]})"
            if label in self.labels_descriptions
            else ""
        )
        template = f"Extract {label}{description} from text"

        examples = [
            {
                "conversations": [
                    {"from": "human", "value": f"Text: {text}"},
                    {"from": "gpt", "value": "I've read this text."},
                    {"from": "human", "value": template},
                    {"from": "gpt", "value": "[]"},
                ]
            }
        ]

        spans = []
        outputs = self.inference(examples=examples)
        for output in outputs:
            if "[" in output and "]" in output:
                try:
                    output = literal_eval(output)
                except:
                    output = [output.strip("[] ")]
            else:
                output = [output]

            for predicted_text in output:
                if predicted_text in text:
                    spans.append(
                        Span(
                            start=text.index(predicted_text),
                            end=text.index(predicted_text) + len(predicted_text),
                            text=predicted_text,
                            label=label,
                        )
                    )
        spans = sorted(spans, key=lambda x: (x.start, x.end))
        return spans

    def _extract(self, text: str, **kwargs: Any) -> Sequence[Span]:
        spans = []
        for label in self.labels:
            spans.extend(self._one_shot(text, label=label))
        return spans
