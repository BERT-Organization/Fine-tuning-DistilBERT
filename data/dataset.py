from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset


@dataclass(frozen=True)
class TextExample:
    text: str
    label: int


class TextClassificationDataset(Dataset):
    def __init__(self, examples: list[TextExample], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = example.label
        return item
