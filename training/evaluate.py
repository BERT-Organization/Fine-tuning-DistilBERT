from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """
    Chạy evaluation loop, trả về loss và accuracy.

    Returns:
        {"loss": float, "accuracy": float}
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
            outputs = model(**batch)

            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss":     total_loss / max(1, len(loader)),
        "accuracy": correct / max(1, total),
    }
