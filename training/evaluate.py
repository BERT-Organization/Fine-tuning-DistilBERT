from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """
    Evaluation cho extractive QA.

    Returns:
        {"loss": float, "span_exact_match": float}
    """
    model.eval()
    total_loss = 0.0
    exact_match = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device=device) for k, v in batch.items() if hasattr(v, "to")}
            outputs = model(**batch)

            total_loss += outputs.loss.item()

            start_pred = outputs.start_logits.argmax(dim=-1)
            end_pred = outputs.end_logits.argmax(dim=-1)
            exact_match += (
                (start_pred == batch["start_positions"]) &
                (end_pred == batch["end_positions"])
            ).sum().item()
            total += batch["start_positions"].size(dim=0)

    return {
        "loss": total_loss / max(1, len(loader)),
        "span_exact_match": exact_match / max(1, total),
    }
