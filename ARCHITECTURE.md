# Kiến trúc dự án

Dự án triển khai pipeline fine-tuning DistilBERT cho extractive QA tiếng Việt. Mô hình không sinh câu trả lời mới; nó chọn một span trong context bằng cách dự đoán token bắt đầu và token kết thúc.

## Luồng tổng thể

```text
Dataset raw
  -> load_raw_datasets()
  -> prepare_train_features() / prepare_eval_features()
  -> tokenizer + sliding window + offset_mapping
  -> input_ids, attention_mask, start_positions, end_positions
  -> DistilBERT encoder
  -> QA classifier
  -> start_logits, end_logits
  -> loss / checkpoint / ONNX export
```

## 1. Data pipeline

### `data/data_loader.py`

Phụ trách tải dữ liệu:

- HuggingFace Hub qua `dataset_name`.
- File local `.json`, `.jsonl`, `.csv`, `.tsv`.
- Tạo `DatasetDict` cho các split `train`, `validation`, `test` nếu có.
- Gọi preprocessing tương ứng cho train/eval.

### `data/dataset.py`

Phụ trách biến raw examples thành features:

- Tokenize question/context bằng tokenizer fast của HuggingFace.
- Dùng `return_overflowing_tokens=True` để tạo nhiều window cho context dài.
- Dùng `return_offsets_mapping=True` để map vị trí ký tự sang token.
- Gán `start_positions` và `end_positions` cho training.
- Mẫu không có đáp án hoặc đáp án bị cắt khỏi window được gán về CLS token.

Segmentation tiếng Việt được hỗ trợ trực tiếp trong preprocessing QA. Khi `underthesea` thêm dấu `_` vào từ ghép, pipeline dùng `align_segmentation_offset` để map offset từ raw context sang segmented context trước khi gán nhãn token span.

### `data/vietnamese_utils.py`

Chứa tiện ích xử lý tiếng Việt, gồm hàm `align_segmentation_offset` để căn chỉnh offset sau segmentation và các helper mapping span.

## 2. Model

### `model/modeling.py`

`DistilBertForQuestionAnswering` gồm:

- `DistilBertModel` pretrained encoder.
- Dropout.
- Linear layer `hidden_size -> 2`.
- Tách output thành `start_logits` và `end_logits`.
- Tính loss bằng trung bình của start loss và end loss.

Forward pass:

```text
input_ids, attention_mask
  -> DistilBERT
  -> last_hidden_state
  -> dropout
  -> linear(hidden_size, 2)
  -> start_logits, end_logits
```

Model có `save_pretrained()` tùy chỉnh để lưu:

- `pytorch_model.bin`
- `config.json`

### `model/config.py`

Tạo `DistilBertConfig` từ checkpoint HuggingFace và áp dụng các tham số như dropout/`num_labels`.

### `model/qa_head.py`

Chứa các hàm/phần mở rộng cho post-processing answer span. Luồng model chính hiện dùng linear QA classifier trực tiếp trong `modeling.py`.

## 3. Training

### `training/config.py`

`TrainingConfig` gom toàn bộ tham số:

- Model: `model_name`, `dropout`, `freeze_encoder`.
- Dataset: `dataset_name`, `train_file`, `validation_file`, tên cột.
- Preprocessing: `max_length`, `doc_stride`, `padding`.
- Training: `batch_size`, `epochs`, `learning_rate`, `weight_decay`.
- Tối ưu GPU: `use_amp`, `use_tf32`, `num_workers`.
- Output/checkpoint: `output_dir`, `save_best_model`, `eval_strategy`.

### `training/trainer.py`

Training loop:

1. Set seed.
2. Chọn CUDA nếu khả dụng, trừ khi bật `force_cpu`.
3. Load tokenizer và dataset.
4. Build dataloader.
5. Build model từ pretrained DistilBERT.
6. Build optimizer và scheduler.
7. Train từng epoch với AMP và gradient accumulation.
8. Tính validation loss nếu có validation labels.
9. Chạy thêm validation pass để tính EM/F1 từ logits bằng `evaluate.load("squad")`.
10. Lưu `best_model` theo `f1` cao nhất và `checkpoint-epoch-*`.

`best_model` hiện dựa trên `f1` cao nhất; cuối vòng train sẽ nạp lại best checkpoint nếu `load_best_model` bật.

### `training/qa_metrics.py`

Cung cấp Exact Match và F1 theo phong cách SQuAD. Trainer đã tích hợp trực tiếp `compute_metrics(eval_preds)` để tính EM/F1 trong validation loop.

## 4. Export và inference

### `scripts/export_onnx.py`

Load checkpoint PyTorch từ `pytorch_model.bin`, export sang:

- `model.onnx`
- `model_quantized.onnx`

Script cũng lưu tokenizer vào `output_dir` để inference có thể load trực tiếp.

### `scripts/inference_onnx.py`

Inference demo:

1. Load tokenizer từ thư mục ONNX.
2. Tokenize question/context.
3. Load `model_quantized.onnx` nếu có, fallback sang `model.onnx`.
4. Lấy argmax của start/end logits.
5. Decode token span thành text.

Đây là demo tối giản. Với production, nên thêm post-processing chọn span hợp lệ tốt nhất thay vì chỉ lấy argmax độc lập.

## 5. Quyết định thiết kế quan trọng

- Dùng `distilbert-base-multilingual-cased` để có pretrained multilingual encoder phù hợp tiếng Việt.
- Dùng thuật toán align offset để vừa tận dụng segmentation vừa giữ đúng vị trí answer span.
- Dùng sliding window để không bỏ context dài.
- Lưu checkpoint theo epoch để dễ rollback/thử nghiệm.
- ONNX + dynamic quantization giúp giảm kích thước và chạy CPU thuận tiện hơn.
