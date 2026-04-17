from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ViQuADExample:
    """Một cặp (question, context) từ UIT-ViQuAD2.0 dùng cho answerability classification."""
    question: str
    context: str
    label: int  # 0 = answerable, 1 = unanswerable (is_impossible)


class ViQuADDataset(Dataset):
    """
    Dataset cho bài toán answerability classification trên UIT-ViQuAD2.0.

    Input model: [CLS] question [SEP] context [SEP]  (chuẩn BERT/DistilBERT cho QA)
    Label      : is_impossible → 0 (có đáp án) / 1 (không có đáp án)
    """

    def __init__(self, examples: list[ViQuADExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.question,
            example.context,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(example.label, dtype=torch.long)
        return item
