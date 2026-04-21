from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.data_loader import build_qa_datasets
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
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"device: {device}")
    print(f"config: {config}\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    datasets = build_qa_datasets(tokenizer, config)
    train_dataset = datasets["train"]
    valid_dataset = datasets["validation"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(config.model_name, config.dropout, config.freeze_encoder)
    model.to(device)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer = build_optimizer(model, config)
    grad_accum_steps = max(1, config.gradient_accumulation_steps)
    total_steps = ((len(train_loader) + grad_accum_steps - 1) // grad_accum_steps) * config.epochs
    scheduler = build_scheduler(optimizer, total_steps, config)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_valid_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if hasattr(v, "to")
            }

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += outputs.loss.item()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # ── Evaluation ─────────────────────────────────────────────────────────
        metrics = evaluate(model, valid_loader, device)

        print(
            f"epoch={epoch}/{config.epochs}  "
            f"train_loss={train_loss / len(train_loader):.4f}  "
            f"valid_loss={metrics['loss']:.4f}  "
            f"valid_span_em={metrics['span_exact_match']:.4f}"
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


