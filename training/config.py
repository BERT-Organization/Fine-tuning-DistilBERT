from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainingConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    model_name:     str   = "distilbert-base-uncased"
    dropout:        float = 0.1
    freeze_encoder: bool  = False   # True = chỉ train QA head

    # ── Data ───────────────────────────────────────────────────────────────────
    dataset_name:         Optional[str] = "taidng/UIT-ViQuAD2.0"
    dataset_config_name:  Optional[str] = None
    train_file:           Optional[str] = None
    validation_file:      Optional[str] = None
    test_file:            Optional[str] = None
    question_column:      str = "question"
    context_column:       str = "context"
    answers_column:       str = "answers"
    impossible_column:    str = "is_impossible"
    plausible_answers_column: str = "plausible_answers"
    max_length:           int = 384
    doc_stride:           int = 128
    padding:              str = "max_length"
    cache_dir:            Optional[str] = None  # HuggingFace dataset cache

    # ── Training loop ──────────────────────────────────────────────────────────
    batch_size:    int   = 8
    epochs:        int   = 3
    learning_rate: float = 2e-5
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.1    # tỉ lệ warmup steps / total steps
    max_grad_norm: float = 1.0    # gradient clipping

    # ── GPU / throughput ──────────────────────────────────────────────────────
    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = True                 # mixed precision với torch.cuda.amp
    gradient_accumulation_steps: int = 1  # batch hiệu dụng = batch_size × steps

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: str = "outputs/checkpoints"
    seed:       int = 42

    # ── Điều chỉnh fine-tuning sau này ─────────────────────────────────────────
    # Thêm field mới tại đây mà không cần chỉnh trainer.py:
    #   label_smoothing: float = 0.0
    #   bf16: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load config từ file YAML."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, path: str | Path) -> None:
        """Lưu config ra file YAML."""
        import dataclasses
        Path(path).write_text(
            yaml.dump(data=dataclasses.asdict(self), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
