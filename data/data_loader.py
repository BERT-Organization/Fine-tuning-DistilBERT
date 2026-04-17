from __future__ import annotations

from datasets import load_dataset, DatasetDict

from .dataset import ViQuADDataset, ViQuADExample


DATASET_NAME = "taidng/UIT-ViQuAD2.0"


def _hf_split_to_examples(hf_split) -> list[ViQuADExample]:
    """Chuyển một HuggingFace split thành list ViQuADExample."""
    examples: list[ViQuADExample] = []
    for row in hf_split:
        examples.append(
            ViQuADExample(
                question=row["question"],
                context=row["context"],
                label=int(row["is_impossible"]),  # False→0, True→1
            )
        )
    return examples


def load_viquad(cache_dir: str | None = None) -> DatasetDict:
    """Tải UIT-ViQuAD2.0 từ HuggingFace Hub."""
    return load_dataset(DATASET_NAME, cache_dir=cache_dir)


def build_dataset(
    split: str,
    tokenizer,
    max_length: int = 512,
    cache_dir: str | None = None,
) -> ViQuADDataset:
    """
    Tải split ("train" / "validation") của UIT-ViQuAD2.0 và trả về ViQuADDataset.

    Args:
        split     : "train" hoặc "validation"
        tokenizer : HuggingFace tokenizer
        max_length: độ dài tối đa (default 512 vì context dài)
        cache_dir : thư mục cache HuggingFace (optional)
    """
    hf_data = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir)
    examples = _hf_split_to_examples(hf_data)
    return ViQuADDataset(examples=examples, tokenizer=tokenizer, max_length=max_length)
