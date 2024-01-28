#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict

from weak_annotators.utils import DictifyMixin


@dataclass
class Span(DictifyMixin):
    start: int
    end: int
    text: str
    label: Optional[str] = None

    def __str__(self):
        return self.text


class BaseExtractor:

    def __init__(
            self,
            labels: List[str],
            labels_descriptions: Optional[Dict[str, str]] = None,
            prompt_template: Optional[str] = None,
            **kwargs
    ):
        self.labels = labels
        self.labels_descriptions = labels_descriptions
        self.prompt_template = prompt_template

    def _extract(self, text, *args, **kwargs) -> Sequence[Span]:
        raise NotImplementedError("This method should be implemented in a child class")

    def _remove_duplicates(self, spans: Sequence[Span]) -> Sequence[Span]:
        """
        Remove duplicate spans
        :param spans:
        :return:
        """
        spans = sorted(spans, key=lambda span: (span.start, span.end))
        spans = [span for i, span in enumerate(spans) if i == 0 or span != spans[i - 1]]
        return spans

    def _post_process(self, spans: Sequence[Span]) -> Sequence[Span]:
        spans = self._remove_duplicates(spans)
        return spans

    def __call__(self, *args, return_dict=False, **kwargs) -> Sequence[Span]:
        spans = self._extract(*args, **kwargs)
        spans = self._post_process(spans)
        if return_dict:
            return [span.to_dict() for span in spans]
        else:
            return spans
