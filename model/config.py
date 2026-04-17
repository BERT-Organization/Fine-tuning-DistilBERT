from __future__ import annotations

from transformers import AutoConfig
from transformers import DistilBertConfig as HFDistilBertConfig


def build_config(model_name: str, num_labels: int) -> HFDistilBertConfig:
    """Load HuggingFace DistilBertConfig từ pretrained model name."""
    return AutoConfig.from_pretrained(model_name, num_labels=num_labels)
