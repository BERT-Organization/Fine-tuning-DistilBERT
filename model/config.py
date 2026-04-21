from __future__ import annotations

from transformers import AutoConfig
from transformers import DistilBertConfig as HFDistilBertConfig


def build_config(model_name: str, num_labels: int) -> HFDistilBertConfig:
    """Load HuggingFace DistilBertConfig từ pretrained model name."""
    return AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=num_labels)
