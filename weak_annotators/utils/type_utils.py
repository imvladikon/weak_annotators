#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json


class DictConverterMixin:
    """
    This mixin allows to convert object to dictionary and back
    """

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, json_str, **kwargs):
        return cls.from_dict(json.loads(json_str))


class DictifyMixin(DictConverterMixin):
    """
    This mixin allows to access object attributes as dictionary keys
    and add dictionary-like behavior to the object
    """

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __setitem__(self, key, value):
        if key not in self.__dict__:
            raise AttributeError(f"object has no attribute {key}")
        setattr(self, key, value)
