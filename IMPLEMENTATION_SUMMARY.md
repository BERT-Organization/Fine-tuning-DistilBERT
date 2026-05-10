# Tóm tắt triển khai

## Mục tiêu

Dự án chuyển pipeline Fine-tuning DistilBERT sang bài toán **hỏi đáp trích xuất tiếng Việt**. Mô hình nhận `question` và `context`, sau đó dự đoán span câu trả lời nằm trong context.

## Thành phần đã có

### Cấu hình

- `config/defaults.yaml`: cấu hình mặc định cho DistilBERT QA.
- `training/config.py`: `TrainingConfig` dạng dataclass, hỗ trợ load/save YAML.
- Các tham số chính: model, dataset, max length, doc stride, batch size, AMP/TF32, checkpoint.

### Data pipeline

- `data/data_loader.py`: load dataset từ HuggingFace Hub hoặc file local.
- `data/dataset.py`: tokenize question/context, xử lý sliding window, tạo `start_positions` và `end_positions`.
- `data/vietnamese_utils.py`: tiện ích xử lý tiếng Việt cho các hướng mở rộng.

Điểm quan trọng: preprocessing QA đã bật lại segmentation và dùng `align_segmentation_offset` để cập nhật `answers.answer_start`/`answers.text` theo segmented context.

### Model

- `model/modeling.py`: `DistilBertForQuestionAnswering`.
- Encoder: `DistilBertModel` từ HuggingFace.
- Head: linear classifier dự đoán 2 logit/token.
- Loss: trung bình cross-entropy cho start và end position.
- Có thể freeze encoder bằng `freeze_encoder`.

### Training

- `scripts/train.py`: entry point CLI.
- `training/trainer.py`: training loop với CUDA/CPU fallback, AMP, TF32, gradient accumulation, scheduler, checkpoint.
- `training/optimizer.py`: optimizer và learning-rate scheduler.
- `training/qa_metrics.py`: Exact Match và F1 cho đánh giá QA, tích hợp `evaluate.load("squad")` qua `compute_metrics(eval_preds)`.

### Export và inference

- `scripts/export_onnx.py`: export checkpoint sang ONNX và tạo bản int8 quantized.
- `scripts/export_onnx.py` hỗ trợ tắt quantization bằng `--no-quantize`.
- `scripts/inference_onnx.py`: demo inference ONNX Runtime với post-processing top-k span search.
- `scripts/flatten_squad.py`: convert dữ liệu SQuAD JSON lồng sang JSONL flatten.
- `tests/test_pipeline_unittest.py`: unit tests cho flatten + answer span preprocessing.

### Tài liệu

- `README.md`: tổng quan và quick start.
- `USAGE.md`: hướng dẫn chạy train/export/inference.
- `DATA_FORMAT.md`: schema dữ liệu hỗ trợ.
- `ARCHITECTURE.md`: kiến trúc và luồng xử lý.
- `COMPLETION_CHECKLIST.md`: checklist trạng thái dự án.

## Trạng thái hiện tại

Dự án đã có pipeline chính cho:

- Fine-tune trên dataset mặc định `taidng/UIT-ViQuAD2.0`.
- Fine-tune với file local đã flatten.
- Lưu checkpoint theo epoch.
- Lưu best model theo `f1` cao nhất.
- Nạp lại best model cuối training nếu `load_best_model` bật.
- Export ONNX và chạy inference demo.

## Kiểm chứng đã chạy

- `.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v` (PASS 3/3).
- Smoke test `scripts/flatten_squad.py` với SQuAD mẫu (convert thành công, output JSONL hợp lệ).
- Kiểm tra CLI `--help` cho `scripts/export_onnx.py` và `scripts/inference_onnx.py` để xác nhận option mới.
