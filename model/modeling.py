from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForQuestionAnswering


class DistilBertForQuestionAnswering(nn.Module):
    """DistilBERT pretrained cho extractive QA với head start/end positions."""

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "dropout"):
            config.dropout = dropout
        if hasattr(config, "qa_dropout"):
            config.qa_dropout = dropout

        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    def freeze_encoder(self) -> None:
        """Đóng băng encoder, chỉ train QA head."""
        base_model = getattr(self.model, self.model.base_model_prefix, self.model)
        for param in base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Mở băng toàn bộ model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            **kwargs,
        )


def build_model(
    model_name: str,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> DistilBertForQuestionAnswering:
    model = DistilBertForQuestionAnswering(model_name, dropout)
    if freeze_encoder:
        model.freeze_encoder()
    return model


