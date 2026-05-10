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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


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
    
    config = build_config(model_name_or_path)
    model = DistilBertForQuestionAnswering(config=config)
    
    # Load weights từ checkpoint
    checkpoint = torch.load(
        Path(model_path) / "pytorch_model.bin",
        map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create dummy input
    dummy_input_ids = torch.randint(0, 1000, (1, 384))
    dummy_attention_mask = torch.ones((1, 384))
    
    logger.info("Exporting to ONNX...")
    
    onnx_model_path = output_dir / "model.onnx"
    
    # Export
    torch.onnx.export(
        model,
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
        help="Apply int8 quantization",
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
