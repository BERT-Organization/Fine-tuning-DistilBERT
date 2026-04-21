from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import DistilBertConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .config import build_config


class DistilBertEmbeddings(nn.Module):
    """Word + position embeddings (DistilBERT không dùng token_type embeddings)."""

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(num_embeddings=config.max_position_embeddings, embedding_dim=config.dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        x = self.word_embeddings(input=input_ids) + self.position_embeddings(input=position_ids)
        x = self.layer_norm(x)
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Self-Attention."""

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.dropout = nn.Dropout(p=config.attention_dropout)

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

    def _shape(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self._shape(x=self.q_lin(input=hidden_states), batch_size=batch_size, seq_len=seq_len)
        k = self._shape(x=self.k_lin(input=hidden_states), batch_size=batch_size, seq_len=seq_len)
        v = self._shape(x=self.v_lin(input=hidden_states), batch_size=batch_size, seq_len=seq_len)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, 1, T], 1=keep, 0=mask
            mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(input=scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [B, H, T, D]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_lin(input=context)


class FeedForward(nn.Module):
    """Transformer FFN block: dim -> hidden_dim -> dim."""

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(input=self.dropout(input=self.activation(input=self.lin1(input=x))))


class TransformerBlock(nn.Module):
    """DistilBERT encoder layer: MHSA + FFN with residual + layernorm."""

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config=config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.ffn = FeedForward(config=config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(hidden_states=x, attention_mask=attention_mask)
        x = self.sa_layer_norm(input=x + self.dropout(input=attn_out))

        ffn_out = self.ffn(x=x)
        x = self.output_layer_norm(input=x + self.dropout(input=ffn_out))
        return x


class DistilBertEncoder(nn.Module):
    """Stack N transformer blocks của DistilBERT."""

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config=config) for _ in range(config.n_layers)])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x, attention_mask=attention_mask)
        return x


class DistilBertForQuestionAnswering(nn.Module):
    """DistilBERT QA model thuần PyTorch, lấy kiến trúc config từ HuggingFace."""

    def __init__(self, config: DistilBertConfig, dropout: float = 0.1):
        super().__init__()
        self.config = config

        if hasattr(self.config, "dropout"):
            self.config.dropout = dropout
        if hasattr(self.config, "qa_dropout"):
            self.config.qa_dropout = dropout

        self.embeddings = DistilBertEmbeddings(config=self.config)
        self.encoder = DistilBertEncoder(config=self.config)

        qa_dropout = getattr(self.config, "qa_dropout", dropout)
        self.dropout = nn.Dropout(p=qa_dropout)
        self.qa_outputs = nn.Linear(in_features=self.config.dim, out_features=2)

        self._init_weights()

    def _init_weights(self) -> None:
        init_std = getattr(self.config, "initializer_range", 0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def freeze_encoder(self) -> None:
        """Đóng băng encoder, chỉ train QA head."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Mở băng toàn bộ model."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        x = self.embeddings(input_ids=input_ids)
        sequence_output = self.encoder(x=x, attention_mask=attention_mask)
        sequence_output = self.dropout(input=sequence_output)
        logits = self.qa_outputs(input=sequence_output)  # [B, T, 2]
        start_logits, end_logits = logits.split(split_size=1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # [B, T]
        end_logits = end_logits.squeeze(-1).contiguous()      # [B, T]

        loss = None
        if start_positions is not None and end_positions is not None:
            if start_positions.dim() > 1:
                start_positions = start_positions.squeeze(-1)
            if end_positions.dim() > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(input=start_logits, target=start_positions)
            end_loss = loss_fct(input=end_logits, target=end_positions)
            loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=None,
            attentions=None,
        )


def build_model(
    model_name: str,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> DistilBertForQuestionAnswering:
    hf_config = build_config(model_name=model_name, num_labels=2)
    model = DistilBertForQuestionAnswering(config=hf_config, dropout=dropout)
    if freeze_encoder:
        model.freeze_encoder()
    return model


