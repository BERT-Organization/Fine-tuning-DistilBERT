"""
Export DistilBERT QA model to ONNX format dengan int8 quantization.

Công dụng:
1. Convert PyTorch model → ONNX format
2. Apply int8 quantization (giảm model size ~4x)
3. Optimize cho inference trên CPU
"""

import logging
import sys
from pathlib import Path
import argparse
import datetime as dt
import json
import subprocess

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ONNXQuestionAnsweringWrapper(nn.Module):
    """Return only tensor outputs so torch.onnx.export can trace cleanly."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        additive_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=torch.float32)) * -10000.0
        outputs = self.model(input_ids=input_ids, attention_mask=additive_mask)
        return outputs.start_logits, outputs.end_logits


def _load_checkpoint_weights(checkpoint_path: Path) -> dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        logger.warning("Current torch version does not support weights_only=True; falling back to default torch.load.")
        return torch.load(checkpoint_path, map_location="cpu")


def _read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            text=True,
        )
        commit = output.strip()
        return commit if commit else None
    except Exception:
        return None


def _write_model_metadata(
    output_dir: Path,
    model_name_or_path: str,
    checkpoint_dir: Path,
    quantized: bool,
    max_length: int,
) -> None:
    onnx_path = output_dir / "model.onnx"
    quantized_path = output_dir / "model_quantized.onnx"
    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer_config = output_dir / "tokenizer_config.json"
    config_json = checkpoint_dir / "config.json"

    metadata = {
        "model_name": model_name_or_path,
        "export_time_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": _resolve_git_commit(),
        "checkpoint_dir": str(checkpoint_dir),
        "quantized": quantized,
        "onnx": {
            "model_path": str(onnx_path),
            "model_size_bytes": onnx_path.stat().st_size if onnx_path.exists() else None,
            "quantized_model_path": str(quantized_path) if quantized_path.exists() else None,
            "quantized_model_size_bytes": quantized_path.stat().st_size if quantized_path.exists() else None,
        },
        "tokenizer": {
            "tokenizer_json_path": str(tokenizer_json),
            "tokenizer_config_path": str(tokenizer_config),
        },
        "runtime": {
            "max_length": max_length,
            "opset_version": 14,
        },
        "checkpoint_config": _read_json_file(config_json),
    }

    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("✓ Metadata saved to %s", metadata_path)


def export_to_onnx(
    model_path: str,
    output_dir: str = "outputs/onnx",
    model_name_or_path: str = "distilbert-base-multilingual-cased",
    quantize: bool = True,
) -> None:
    """
    Export DistilBERT QA model to ONNX format.
    
    Args:
        model_path: Path to fine-tuned model (PyTorch checkpoint directory)
        output_dir: Directory để lưu ONNX model
        model_name_or_path: HuggingFace model name cho tokenizer
        quantize: Có apply int8 quantization
    """
    
    logger.info(f"Loading model from {model_path}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    from model.modeling import DistilBertForQuestionAnswering
    from model.config import build_config

    checkpoint_dir = Path(model_path)
    config = build_config(model_name_or_path)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    model = DistilBertForQuestionAnswering(config=config)
    
    # Load weights từ checkpoint
    checkpoint = _load_checkpoint_weights(checkpoint_dir / "pytorch_model.bin")
    model.load_state_dict(checkpoint)
    model.eval()
    onnx_model = ONNXQuestionAnsweringWrapper(model)
    onnx_model.eval()
    
    # Create dummy input
    dummy_input_ids = torch.randint(0, config.vocab_size, (1, 384), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 384), dtype=torch.long)
    
    logger.info("Exporting to ONNX...")
    
    onnx_model_path = output_dir / "model.onnx"
    
    # Export
    torch.onnx.export(
        onnx_model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["start_logits", "end_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "start_logits": {0: "batch_size", 1: "seq_length"},
            "end_logits": {0: "batch_size", 1: "seq_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    logger.info(f"✓ ONNX model saved to {onnx_model_path}")
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model verified")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
    
    # Apply quantization
    if quantize:
        logger.info("Applying int8 quantization...")
        
        quantized_model_path = output_dir / "model_quantized.onnx"
        
        quantize_dynamic(
            str(onnx_model_path),
            str(quantized_model_path),
            weight_type=QuantType.QInt8,
        )
        
        logger.info(f"✓ Quantized model saved to {quantized_model_path}")
        
        # Compare sizes
        orig_size = onnx_model_path.stat().st_size / (1024**2)
        quant_size = quantized_model_path.stat().st_size / (1024**2)
        compression = (1 - quant_size / orig_size) * 100
        
        logger.info(f"Original size: {orig_size:.2f} MB")
        logger.info(f"Quantized size: {quant_size:.2f} MB")
        logger.info(f"Compression: {compression:.1f}%")
    
    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✓ Tokenizer saved to {output_dir}")

    _write_model_metadata(
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint_dir,
        quantized=quantize,
        max_length=384,
    )


def main():
    parser = argparse.ArgumentParser(description="Export DistilBERT QA to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/checkpoints/best_model",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply int8 quantization (default: enabled)",
    )
    parser.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_false",
        help="Disable int8 quantization",
    )
    
    args = parser.parse_args()
    
    export_to_onnx(
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
