from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.data_loader import build_dataset
from model.modeling import build_model
from .config import TrainingConfig
from .evaluate import evaluate
from .optimizer import build_optimizer, build_scheduler


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"config: {config}\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = build_dataset("train",      tokenizer, config.max_length, config.cache_dir)
    valid_dataset = build_dataset("validation", tokenizer, config.max_length, config.cache_dir)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(config.model_name, config.num_labels, config.dropout, config.freeze_encoder)
    model.to(device)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * config.epochs
    scheduler = build_scheduler(optimizer, total_steps, config)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_valid_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += outputs.loss.item()

        # ── Evaluation ─────────────────────────────────────────────────────────
        metrics = evaluate(model, valid_loader, device)

        print(
            f"epoch={epoch}/{config.epochs}  "
            f"train_loss={train_loss / len(train_loader):.4f}  "
            f"valid_loss={metrics['loss']:.4f}  "
            f"valid_acc={metrics['accuracy']:.4f}"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────────
        ckpt_dir = Path(config.output_dir) / f"epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        tokenizer.save_pretrained(ckpt_dir)
        config.to_yaml(ckpt_dir / "config.yaml")

        # Lưu riêng best model
        if metrics["loss"] < best_valid_loss:
            best_valid_loss = metrics["loss"]
            best_dir = Path(config.output_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_dir / "model.pt")
            tokenizer.save_pretrained(best_dir)
            config.to_yaml(best_dir / "config.yaml")
            print(f"  → best model saved (valid_loss={best_valid_loss:.4f})")


