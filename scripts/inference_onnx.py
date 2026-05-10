"""
Inference demo using ONNX model.

Chạy end-to-end inference (preprocessing, model, post-processing) với ONNX model.
"""

import logging
from pathlib import Path
import argparse
import numpy as np

from transformers import AutoTokenizer
import onnxruntime as ort

logger = logging.getLogger(__name__)


def run_inference_onnx(
    model_dir: str,
    question: str,
    context: str,
    max_length: int = 384,
) -> dict:
    """
    Run inference using ONNX model.
    
    Args:
        model_dir: Directory chứa ONNX model + tokenizer
        question: Input question
        context: Input context
        max_length: Max sequence length
        
    Returns:
        Dict chứa answer_text, start_pos, end_pos, scores
    """
    
    model_dir = Path(model_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    # Tokenize
    logger.info(f"Question: {question}")
    logger.info(f"Context: {context[:100]}...")
    
    encoding = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    
    input_ids = encoding["input_ids"].astype(np.int64)
    attention_mask = encoding["attention_mask"].astype(np.int64)
    
    # Load ONNX model
    model_path = model_dir / "model_quantized.onnx"
    if not model_path.exists():
        model_path = model_dir / "model.onnx"
    
    logger.info(f"Loading ONNX model from {model_path}...")
    
    session = ort.InferenceSession(str(model_path))
    
    # Run inference
    logger.info("Running inference...")
    
    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    
    start_logits = outputs[0][0]  # (seq_len,)
    end_logits = outputs[1][0]    # (seq_len,)
    
    # Extract answer
    start_idx = start_logits.argmax()
    end_idx = end_logits.argmax()
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    # Limit answer length
    if end_idx - start_idx + 1 > 30:
        end_idx = start_idx + 29
    
    # Decode tokens to text
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer_tokens = tokens[start_idx:end_idx + 1]
    answer_text = tokenizer.convert_tokens_to_string(answer_tokens)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Answer: {answer_text}")
    logger.info(f"Start position: {start_idx}")
    logger.info(f"End position: {end_idx}")
    logger.info(f"Start score: {start_logits[start_idx]:.4f}")
    logger.info(f"End score: {end_logits[end_idx]:.4f}")
    logger.info(f"{'='*60}\n")
    
    return {
        "answer": answer_text,
        "start_pos": int(start_idx),
        "end_pos": int(end_idx),
        "start_score": float(start_logits[start_idx]),
        "end_score": float(end_logits[end_idx]),
    }


def main():
    parser = argparse.ArgumentParser(description="ONNX inference demo")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/onnx",
        help="Directory containing ONNX model",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Hà Nội là thủ đô của nước nào?",
        help="Input question",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="Hà Nội là thủ đô của Việt Nam. Thành phố nằm ở phía Bắc Việt Nam.",
        help="Input context",
    )
    
    args = parser.parse_args()
    
    result = run_inference_onnx(
        model_dir=args.model_dir,
        question=args.question,
        context=args.context,
    )
    
    print(f"\nResult: {result}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
