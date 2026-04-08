from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data.data_loader import build_dataset
from model.modeling import build_model


@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    train_path: str = "data/train.tsv"
    valid_path: str = "data/valid.tsv"
    output_dir: str = "outputs/checkpoints"
    max_length: int = 256
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    num_labels: int = 2
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = build_dataset(config.train_path, tokenizer, config.max_length)
    valid_dataset = build_dataset(config.valid_path, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    model = build_model(config.model_name, config.num_labels)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, total_steps // 10),
        num_training_steps=total_steps,
    )

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items() if hasattr(value, "to")}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {key: value.to(device) for key, value in batch.items() if hasattr(value, "to")}
                outputs = model(**batch)
                valid_loss += outputs.loss.item()

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir / f"epoch-{epoch + 1}")
        tokenizer.save_pretrained(output_dir / f"epoch-{epoch + 1}")

        print(
            f"epoch={epoch + 1} train_loss={train_loss / max(1, len(train_loader)):.4f} "
            f"valid_loss={valid_loss / max(1, len(valid_loader)):.4f}"
        )
