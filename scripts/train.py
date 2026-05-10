from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.config import TrainingConfig
from training.trainer import train


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT cho Extractive QA trên Vietnamese datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/defaults.yaml",
        help="Path to YAML config file",
    )
    # CLI overrides
    parser.add_argument("--model_name", type=str, help="Model name/checkpoint")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--max_length", type=int, help="Max sequence length")
    parser.add_argument("--doc_stride", type=int, help="Doc stride for sliding window")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--dataset_name", type=str, help="Dataset name (HF Hub)")
    parser.add_argument("--train_file", type=str, help="Path to train file")
    parser.add_argument("--validation_file", type=str, help="Path to validation file")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU training even if CUDA is available")
    
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Main training entry point."""
    
    args = parse_args(argv=argv)

    # Load config từ YAML
    config_path = Path(args.config)
    if config_path.exists():
        config = TrainingConfig.from_yaml(str(config_path))
    else:
        print(f"Config file not found: {config_path}. Using defaults.")
        config = TrainingConfig()

    # Override từ CLI arguments
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None
    }
    
    for key, value in cli_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config key: {key}")

    # Start training
    train(config=config)


if __name__ == "__main__":
    main()


