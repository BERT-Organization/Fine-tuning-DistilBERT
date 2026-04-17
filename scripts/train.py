from __future__ import annotations

import argparse
from pathlib import Path

from training.config import TrainingConfig
from training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on UIT-ViQuAD2.0")
    parser.add_argument(
        "--config",
        type=str,
        default="config/defaults.yaml",
        help="Đường dẫn đến file YAML config (mặc định: config/defaults.yaml)",
    )
    # Cho phép override bất kỳ field nào qua CLI, vd: --learning_rate 3e-5
    parser.add_argument("--model_name",     type=str)
    parser.add_argument("--epochs",         type=int)
    parser.add_argument("--batch_size",     type=int)
    parser.add_argument("--learning_rate",  type=float)
    parser.add_argument("--max_length",     type=int)
    parser.add_argument("--dropout",        type=float)
    parser.add_argument("--output_dir",     type=str)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--seed",           type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config từ YAML
    config_path = Path(args.config)
    config = TrainingConfig.from_yaml(config_path) if config_path.exists() else TrainingConfig()

    # Override từ CLI nếu được cung cấp
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None
    }
    for key, value in cli_overrides.items():
        setattr(config, key, value)

    train(config)


if __name__ == "__main__":
    main()

