
## Weak Annotators (NER)

Experiments with weak annotators for NER using different models and methods including LLMs.

### Installation

```bash
pip install git+https://github.com/imvladikon/weak-annotators.git
```

### Usage

1. Using [`UniversalNER`](https://universal-ner.github.io/) extractor:

```python
from weak_annotators import UniversalNerExtractor

text = """
The patient was prescribed 100 mg of aspirin daily for 3 days.
""".strip()
labels = ["DRUG", "DISEASE", "SYMPTOM", "DURATION"]
extractor = UniversalNerExtractor(labels=labels)
print(extractor(text))
# [Span(start=37, end=44, text='aspirin', label='DRUG'), Span(start=55, end=61, text='3 days', label='DURATION')]
```

It returns a list of `Spans` but if pass `return_dict=True` it will return a list of dictionaries:

```python
print(extractor(text, return_dict=True))
# [{'start': 37, 'end': 44, 'text': 'aspirin', 'label': 'DRUG'}, {'start': 55, 'end': 61, 'text': '3 days', 'label': 'DURATION'}]
```

2. Using [`medalpaca`](https://huggingface.co/medalpaca/medalpaca-7b) LLM:

It requires labels descriptions:

```python

from weak_annotators import MedAlpacaExtractor

labels = ["DRUG", "DISEASE", "SYMPTOM", "DURATION"]
labels_descriptions = {
    "DRUG": "Drug or medication",
    "DISEASE": "Any disease, syndrome, or medical condition",
    "SYMPTOM": "Any symptom or sign of a disease or medical condition",
    "DURATION": "Any period of time",
}
extractor = MedAlpacaExtractor(labels=labels, labels_description=labels_descriptions)

text = """
The patient was prescribed 100 mg of aspirin daily for 3 days.
""".strip()

annotations = extractor(text)
print(annotations)

```

Optionally, it's possible to pass `prompt_template` to `MedAlpacaExtractor`.
```python
prompt_template = "Extract entities of type {} from the following text:"
extractor = MedAlpacaExtractor(labels=labels, labels_description=labels_descriptions, prompt_template=prompt_template)
```


3. Using `flair` (TARS extractor):

```python
from weak_annotators import FlairExtractor

labels = ["DRUG", "DISEASE", "SYMPTOM", "DURATION"]
extractor = FlairExtractor(labels=labels)
print(extractor(text))
```

