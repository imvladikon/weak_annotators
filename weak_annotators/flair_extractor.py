#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence, List

from flair.models import TARSTagger
from flair.data import Sentence

from weak_annotators.base_extractor import BaseExtractor, Span
from weak_annotators.model_registry import ModelsRegistry


class FlairExtractor(BaseExtractor):

    def __init__(self, labels: List[str], **kwargs):
        super().__init__(labels=labels, **kwargs)
        task_name = ", ".join(sorted(labels))
        if not ModelsRegistry.has(task_name):
            tagger = TARSTagger.load("tars-ner")
            tagger.add_and_switch_to_new_task(task_name, labels, label_type="ner")
            ModelsRegistry.register_model(task_name, tagger)

        self.tagger = ModelsRegistry.get_model(task_name)

    def _extract(self, text, *args, **kwargs) -> Sequence[Span]:
        sentence = Sentence(text)
        self.tagger.predict(sentence)

        spans = []
        for entity in sentence.get_spans("ner"):
            spans.append(
                Span(
                    start=entity.start_position,
                    end=entity.end_position,
                    text=entity.text,
                    label=entity.get_labels()[0].value,
                )
            )
        return spans
