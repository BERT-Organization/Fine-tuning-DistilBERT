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
    max_answer_length: int = 30,
    n_best_size: int = 20,
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
        return_offsets_mapping=True,
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

    # Advanced span post-processing: top-k start/end pairs với ràng buộc hợp lệ.
    offset_mapping = encoding["offset_mapping"][0]
    start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
    end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]

    best_span = None
    best_score = float("-inf")
    for start_idx in start_indexes:
        for end_idx in end_indexes:
            if end_idx < start_idx:
                continue
            if end_idx - start_idx + 1 > max_answer_length:
                continue

            start_char, _ = offset_mapping[start_idx]
            _, end_char = offset_mapping[end_idx]
            if end_char <= start_char:
                continue

            score = float(start_logits[start_idx] + end_logits[end_idx])
            if score > best_score:
                best_score = score
                best_span = (int(start_idx), int(end_idx), int(start_char), int(end_char))

    if best_span is None:
        start_idx = int(start_logits.argmax())
        end_idx = start_idx
        answer_text = ""
        span_score = float(start_logits[start_idx] + end_logits[end_idx])
    else:
        start_idx, end_idx, start_char, end_char = best_span
        answer_text = context[start_char:end_char].strip()
        span_score = best_score
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Answer: {answer_text}")
    logger.info(f"Start position: {start_idx}")
    logger.info(f"End position: {end_idx}")
    logger.info(f"Start score: {start_logits[start_idx]:.4f}")
    logger.info(f"End score: {end_logits[end_idx]:.4f}")
    logger.info(f"Span score: {span_score:.4f}")
    logger.info(f"{'='*60}\n")
    
    return {
        "answer": answer_text,
        "start_pos": int(start_idx),
        "end_pos": int(end_idx),
        "start_score": float(start_logits[start_idx]),
        "end_score": float(end_logits[end_idx]),
        "span_score": span_score,
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
    parser.add_argument("--max_answer_length", type=int, default=30, help="Maximum answer token length")
    parser.add_argument("--n_best_size", type=int, default=20, help="Top-k start/end candidates for span search")
    
    args = parser.parse_args()
    
    result = run_inference_onnx(
        model_dir=args.model_dir,
        question=args.question,
        context=args.context,
        max_answer_length=args.max_answer_length,
        n_best_size=args.n_best_size,
    )
    
    print(f"\nResult: {result}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
