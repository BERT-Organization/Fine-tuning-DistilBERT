# Fine-tuning DistilBERT cho hỏi đáp trích xuất tiếng Việt

Dự án này fine-tune **DistilBERT multilingual cased** cho bài toán **extractive question answering**: mô hình nhận một câu hỏi và một đoạn ngữ cảnh, sau đó dự đoán vị trí bắt đầu/kết thúc của câu trả lời nằm trực tiếp trong ngữ cảnh.

## Tính năng chính

- Fine-tune `distilbert-base-multilingual-cased` với QA head dự đoán `start_logits` và `end_logits`.
- Hỗ trợ dataset HuggingFace Hub, mặc định là `taidng/UIT-ViQuAD2.0`.
- Hỗ trợ file local dạng JSON/JSONL/CSV/TSV theo schema QA tương thích SQuAD.
- Tokenization với sliding window (`max_length`, `doc_stride`) để xử lý context dài.
- Bật lại Vietnamese segmentation (`underthesea`) kèm thuật toán align offset để đồng bộ `answer_start` sau khi thêm `_`.
- Xử lý câu hỏi không có đáp án bằng vị trí CLS token.
- Training loop có AMP, TF32, gradient accumulation, gradient clipping và checkpoint theo epoch.
- Validation tích hợp EM/F1 (chuẩn SQuAD qua `evaluate`) và chọn best model theo `f1`.
- Export mô hình sang ONNX và quantization int8 để phục vụ inference CPU.

## Cài đặt

```bash
cd /home/quagntam/Projects/Fine-tuning-DistilBERT
pip install -r requirements.txt
```

Nếu dùng CUDA, hãy bảo đảm phiên bản `torch` trong môi trường phù hợp với driver/GPU hiện tại.

## Chạy nhanh

Train với cấu hình mặc định:

```bash
python scripts/train.py
```

Train với tham số tùy chỉnh:

```bash
python scripts/train.py \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --output_dir outputs/checkpoints/experiment_01
```

Train bằng dataset local:

```bash
python scripts/train.py \
  --dataset_name "" \
  --train_file data/train.json \
  --validation_file data/valid.json
```

Lưu ý: `argparse` nhận `--dataset_name ""` như chuỗi rỗng để bỏ dataset mặc định trong YAML. Cách sạch hơn là tạo một file config riêng với `dataset_name: null`.

## Cấu trúc dự án

```text
Fine-tuning-DistilBERT/
├── config/
│   └── defaults.yaml          # Cấu hình model, dataset, training và output
├── data/
│   ├── data_loader.py         # Load dataset từ HuggingFace hoặc file local
│   ├── dataset.py             # Tokenize, sliding window, tạo start/end labels
│   └── vietnamese_utils.py    # Tiện ích xử lý tiếng Việt
├── model/
│   ├── config.py              # Tạo DistilBertConfig
│   ├── modeling.py            # DistilBERT encoder + QA head
│   └── qa_head.py             # Hàm hậu xử lý span dự đoán
├── training/
│   ├── config.py              # TrainingConfig dataclass
│   ├── trainer.py             # Training loop chính
│   ├── optimizer.py           # Optimizer và scheduler
│   └── qa_metrics.py          # Exact Match và F1 cho QA
├── scripts/
│   ├── train.py               # Entry point fine-tuning
│   ├── export_onnx.py         # Export ONNX và quantization
│   └── inference_onnx.py      # Demo inference bằng ONNX Runtime
└── outputs/                   # Checkpoint và model export sinh ra khi chạy
```

## Cấu hình mặc định

Các tham số chính nằm trong `config/defaults.yaml`:

```yaml
model_name: distilbert-base-multilingual-cased
dataset_name: taidng/UIT-ViQuAD2.0
max_length: 384
doc_stride: 128
batch_size: 16
epochs: 3
learning_rate: 3.0e-5
gradient_accumulation_steps: 1
use_amp: true
output_dir: outputs/checkpoints
```

`use_vietnamese_segmentation` hiện hỗ trợ trực tiếp trong pipeline train. Khi segmentation tạo dấu `_`, hệ thống dùng hàm align offset để cập nhật lại `answers.answer_start` và `answers.text` trước khi tokenize.

## Output khi training

Sau mỗi epoch, trainer lưu checkpoint tại:

```text
outputs/checkpoints/
├── checkpoint-epoch-1/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files...
├── checkpoint-epoch-2/
└── best_model/
    ├── pytorch_model.bin
    ├── config.json
    └── tokenizer files...
```

`best_model` hiện được chọn theo `f1` cao nhất trên validation; cuối train sẽ nạp lại best checkpoint nếu `load_best_model` bật.

## Export ONNX

```bash
python scripts/export_onnx.py \
  --model_path outputs/checkpoints/best_model \
  --output_dir outputs/onnx \
  --model_name distilbert-base-multilingual-cased
```

Script CLI hiện quantize int8 theo mặc định và lưu tokenizer cùng thư mục ONNX.

## Inference ONNX

```bash
python scripts/inference_onnx.py \
  --model_dir outputs/onnx \
  --question "Hà Nội là thủ đô của nước nào?" \
  --context "Hà Nội là thủ đô của Việt Nam. Thành phố nằm ở phía Bắc Việt Nam."
```

## Tài liệu

- [USAGE.md](USAGE.md): hướng dẫn chạy train, export và inference.
- [DATA_FORMAT.md](DATA_FORMAT.md): định dạng dataset được hỗ trợ.
- [ARCHITECTURE.md](ARCHITECTURE.md): kiến trúc và luồng xử lý.
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md): tóm tắt các thành phần đã triển khai.
- [COMPLETION_CHECKLIST.md](COMPLETION_CHECKLIST.md): checklist trạng thái dự án.
