from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from training.config import TrainingConfig
from training.trainer import train


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--doc_stride",     type=int)
    parser.add_argument("--dropout",        type=float)
    parser.add_argument("--output_dir",     type=str)
    parser.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed",           type=int)
    parser.add_argument("--num_workers",    type=int)
    parser.add_argument("--pin_memory",     action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_amp",        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--dataset_name",    type=str)
    parser.add_argument("--dataset_config_name", type=str)
    parser.add_argument("--train_file",      type=str)
    parser.add_argument("--validation_file", type=str)
    parser.add_argument("--test_file",       type=str)
    parser.add_argument("--question_column",  type=str)
    parser.add_argument("--context_column",   type=str)
    parser.add_argument("--answers_column",   type=str)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    config: TrainingConfig | None = None,
) -> None:
    args = parse_args(argv=argv)

    # Load config từ YAML
    if config is None:
        config_path = Path(args.config)
        config = TrainingConfig.from_yaml(path=config_path) if config_path.exists() else TrainingConfig()

    # Override từ CLI nếu được cung cấp
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None
    }
    for key, value in cli_overrides.items():
        setattr(config, key, value)

    train(config=config)


if __name__ == "__main__":
    main()

