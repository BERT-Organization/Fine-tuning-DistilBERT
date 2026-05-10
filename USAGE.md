# Hướng dẫn sử dụng

Tài liệu này mô tả cách chạy fine-tuning, export và inference cho dự án DistilBERT extractive QA tiếng Việt.

## 1. Chuẩn bị môi trường

```bash
cd /home/quagntam/Projects/Fine-tuning-DistilBERT
pip install -r requirements.txt
```

Kiểm tra nhanh môi trường:

```bash
python -c "import torch, transformers, datasets; print(torch.__version__)"
```

## 2. Train với cấu hình mặc định

```bash
python scripts/train.py
```

Cấu hình mặc định nằm ở `config/defaults.yaml`:

- Model: `distilbert-base-multilingual-cased`
- Dataset: `taidng/UIT-ViQuAD2.0`
- `max_length`: 384
- `doc_stride`: 128
- `batch_size`: 16
- `epochs`: 3
- `learning_rate`: `3.0e-5`
- Output: `outputs/checkpoints`

## 3. Train với config riêng

Tạo file ví dụ `config/local.yaml`:

```yaml
model_name: distilbert-base-multilingual-cased
dataset_name: null
train_file: data/train.json
validation_file: data/valid.json

max_length: 384
doc_stride: 128
batch_size: 8
epochs: 5
learning_rate: 2.0e-5
output_dir: outputs/checkpoints/local_run
```

Chạy:

```bash
python scripts/train.py --config config/local.yaml
```

## 4. Override tham số từ CLI

```bash
python scripts/train.py \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 2 \
  --output_dir outputs/checkpoints/run_01
```

Các override đang có trong `scripts/train.py` gồm:

- `--model_name`
- `--epochs`
- `--batch_size`
- `--learning_rate`
- `--max_length`
- `--doc_stride`
- `--dropout`
- `--output_dir`
- `--seed`
- `--num_workers`
- `--gradient_accumulation_steps`
- `--dataset_name`
- `--train_file`
- `--validation_file`
- `--force_cpu`

## 5. Gợi ý cấu hình

Dataset nhỏ hoặc GPU ít VRAM:

```yaml
batch_size: 4
gradient_accumulation_steps: 4
epochs: 5
learning_rate: 2.0e-5
use_amp: true
num_workers: 0
```

GPU ổn định hơn, muốn train nhanh:

```yaml
batch_size: 16
gradient_accumulation_steps: 1
epochs: 3
learning_rate: 3.0e-5
use_amp: true
use_tf32: true
num_workers: 2
```

Chạy CPU để debug:

```bash
python scripts/train.py --force_cpu --batch_size 2 --num_workers 0
```

## 6. Quá trình training

Trainer thực hiện các bước:

1. Load tokenizer và dataset.
2. Tokenize question/context bằng sliding window.
3. Tạo `start_positions` và `end_positions` từ `answers.answer_start`.
4. Load DistilBERT pretrained encoder và QA head.
5. Train với AMP nếu dùng CUDA.
6. Chạy validation loss và tính thêm `exact_match`/`f1` từ logits.
7. Lưu `checkpoint-epoch-*` sau mỗi epoch và cập nhật `best_model` theo `f1` cao nhất.

Output chính:

```text
outputs/checkpoints/
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
├── checkpoint-epoch-3/
└── best_model/
```

Mỗi thư mục checkpoint chứa `pytorch_model.bin`, `config.json` và tokenizer files.

## 7. Export sang ONNX

Sau khi có `best_model`:

```bash
python scripts/export_onnx.py \
  --model_path outputs/checkpoints/best_model \
  --output_dir outputs/onnx \
  --model_name distilbert-base-multilingual-cased
```

Script tạo:

```text
outputs/onnx/
├── model.onnx
├── model_quantized.onnx
└── tokenizer files...
```

CLI hiện bật quantization int8 theo mặc định.

## 8. Inference bằng ONNX Runtime

```bash
python scripts/inference_onnx.py \
  --model_dir outputs/onnx \
  --question "Hà Nội là thủ đô của nước nào?" \
  --context "Hà Nội là thủ đô của Việt Nam. Thành phố nằm ở phía Bắc Việt Nam."
```

Script ưu tiên `model_quantized.onnx`; nếu file này không tồn tại thì dùng `model.onnx`.

## 9. Lưu ý về tách từ tiếng Việt

`config/defaults.yaml` có tham số:

```yaml
use_vietnamese_segmentation: true/false
segmentation_tool: underthesea
```

Trong bài toán extractive QA, segmentation bằng `underthesea` có thể chèn `_` làm lệch offset ký tự. Pipeline hiện đã xử lý bằng thuật toán align (`align_segmentation_offset`) để map `answer_start` từ raw context sang segmented context trước tokenization.

## 10. Xử lý lỗi thường gặp

OOM trên GPU:

```bash
python scripts/train.py --batch_size 4 --gradient_accumulation_steps 4
```

Dataset local không load được:

- Kiểm tra file là JSON/JSONL/CSV/TSV.
- Kiểm tra các cột `question`, `context`, `answers`.
- Với JSON SQuAD lồng `data -> paragraphs -> qas`, xem thêm [DATA_FORMAT.md](DATA_FORMAT.md) vì HuggingFace `load_dataset("json")` có thể cần dữ liệu đã flatten theo dòng/mẫu.

Validation không lưu `best_model`:

- Cần có split `validation`.
- Split validation phải có cột `answers` để trainer tính loss và EM/F1.

Lỗi thiếu thư viện `evaluate` khi validate:

- Cài thêm package: `pip install evaluate`.

ONNX export lỗi không tìm checkpoint:

- Kiểm tra đường dẫn có file `pytorch_model.bin`.
- Dùng đúng `--model_name` tương ứng với tokenizer/model đã train.
