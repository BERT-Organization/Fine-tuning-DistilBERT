from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .dataset import TextClassificationDataset, TextExample


def load_examples_from_tsv(path: str | Path) -> list[TextExample]:
    file_path = Path(path)
    examples: list[TextExample] = []

    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label_text, text = line.split("\t", maxsplit=1)
        examples.append(TextExample(text=text.strip(), label=int(label_text)))

    return examples


def build_dataset(path: str | Path, tokenizer, max_length: int = 256) -> TextClassificationDataset:
    examples = load_examples_from_tsv(path)
    return TextClassificationDataset(examples=examples, tokenizer=tokenizer, max_length=max_length)
