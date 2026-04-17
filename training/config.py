from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainingConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    model_name:     str   = "distilbert-base-uncased"
    num_labels:     int   = 2
    dropout:        float = 0.1
    freeze_encoder: bool  = False   # True = chỉ train classification head

    # ── Data ───────────────────────────────────────────────────────────────────
    max_length: int          = 512
    cache_dir:  Optional[str] = None  # HuggingFace dataset cache

    # ── Training loop ──────────────────────────────────────────────────────────
    batch_size:    int   = 8
    epochs:        int   = 3
    learning_rate: float = 2e-5
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.1    # tỉ lệ warmup steps / total steps
    max_grad_norm: float = 1.0    # gradient clipping

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: str = "outputs/checkpoints"
    seed:       int = 42

    # ── Điều chỉnh fine-tuning sau này ─────────────────────────────────────────
    # Thêm field mới tại đây mà không cần chỉnh trainer.py:
    #   label_smoothing: float = 0.0
    #   fp16: bool = False
    #   gradient_accumulation_steps: int = 1

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
            yaml.dump(dataclasses.asdict(self), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
