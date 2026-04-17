from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class DistilBertForClassification(nn.Module):
    """
    DistilBERT pretrained + custom classification head.

    Architecture (có thể điều chỉnh tự do):
        DistilBERT encoder  →  [CLS] hidden state
        → Dropout
        → Linear (hidden_dim → hidden_dim)       ← layer tuỳ chỉnh 1
        → GELU + Dropout
        → Linear (hidden_dim → num_labels)        ← output layer
        → logits

    Để thêm / bớt layer: chỉnh phần `self.classifier` bên dưới.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        # ── Pretrained DistilBERT encoder (frozen hoặc fine-tune toàn bộ) ──────
        self.distilbert: DistilBertModel = AutoModel.from_pretrained(model_name)
        hidden_dim = self.distilbert.config.hidden_size

        # ── Classification head (tuỳ chỉnh tại đây) ─────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),   # layer tuỳ chỉnh — thêm/bớt ở đây
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    # ── Tiện ích đóng băng / mở băng encoder ─────────────────────────────────

    def freeze_encoder(self) -> None:
        """Đóng băng toàn bộ DistilBERT — chỉ train classification head."""
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Mở băng DistilBERT để fine-tune toàn bộ."""
        for param in self.distilbert.parameters():
            param.requires_grad = True

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """
        Args:
            input_ids      : [B, T]
            attention_mask : [B, T] (optional)
            labels         : [B] long tensor (optional — tính cross-entropy loss)
        Returns:
            SequenceClassifierOutput(loss, logits)
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBERT trả về last_hidden_state; lấy vector [CLS] (index 0)
        cls_output = outputs.last_hidden_state[:, 0, :]   # [B, hidden_dim]
        logits = self.classifier(cls_output)              # [B, num_labels]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_model(
    model_name: str,
    num_labels: int,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> DistilBertForClassification:
    """
    Khởi tạo DistilBertForClassification.

    Args:
        model_name     : HuggingFace model ID (vd: "distilbert-base-uncased")
        num_labels     : số lớp phân loại
        dropout        : dropout rate cho classification head
        freeze_encoder : True = chỉ train head, False = fine-tune toàn bộ
    """
    model = DistilBertForClassification(model_name, num_labels, dropout)
    if freeze_encoder:
        model.freeze_encoder()
    return model


