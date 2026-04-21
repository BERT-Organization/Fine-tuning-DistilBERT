# Fine-tuning DistilBERT

Fine-tune **DistilBERT** cho bài toán **Answerability Classification** trên dataset **UIT-ViQuAD2.0** (tiếng Việt).  
Input: cặp `(question, context)` — Label: `is_impossible` (0 = có đáp án / 1 = không có đáp án).

---

## Cấu trúc dự án

```
Fine-tuning-DistilBERT/
├── config/
│   └── defaults.yaml          # Tất cả hyperparameter — chỉnh tại đây
│
├── data/
│   ├── dataset.py             # ViQuADExample + ViQuADDataset
│   └── data_loader.py         # Load từ HuggingFace Hub (taidng/UIT-ViQuAD2.0)
│
├── model/
│   ├── config.py              # build_config() → HuggingFace DistilBertConfig
│   └── modeling.py            # DistilBertForClassification + build_model()
│
├── training/
│   ├── config.py              # TrainingConfig (đọc/ghi YAML)
│   ├── optimizer.py           # AdamW + Linear warmup scheduler
│   ├── evaluate.py            # evaluate() → {loss, accuracy}
│   └── trainer.py             # Training loop chính
│
├── scripts/
│   └── train.py               # Entrypoint CLI
│
├── requirements.txt
└── outputs/
    └── checkpoints/           # Checkpoint mỗi epoch + best model
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cách chạy

### Chạy với config mặc định

```bash
python scripts/train.py
```

### Chạy với file config tùy chỉnh

```bash
python scripts/train.py --config config/defaults.yaml
```

### Override tham số từ CLI

```bash
# Đổi learning rate và số epoch
python scripts/train.py --learning_rate 3e-5 --epochs 5

# Chỉ train classification head, đóng băng encoder
python scripts/train.py --freeze_encoder

# Đổi output directory
python scripts/train.py --output_dir outputs/experiment_01
```

---

## Cấu hình (`config/defaults.yaml`)

```yaml
# Model
model_name: distilbert-base-uncased
num_labels: 2
dropout: 0.1
freeze_encoder: false   # true = chỉ train classification head

# Data
max_length: 512         # context dài nên để 512

# Training
batch_size: 8
epochs: 3
learning_rate: 2.0e-5
weight_decay: 0.01
warmup_ratio: 0.1
max_grad_norm: 1.0

# Output
output_dir: outputs/checkpoints
seed: 42
```

---

## Điều chỉnh fine-tuning

| Muốn thay đổi | Chỉnh ở đâu |
|---|---|
| Hyperparameter | `config/defaults.yaml` |
| Thêm / bớt layer trong classification head | `model/modeling.py` — `self.classifier` |
| Đổi optimizer hoặc scheduler | `training/optimizer.py` |
| Thêm metric (F1, precision, recall) | `training/evaluate.py` |
| Gradient accumulation, fp16, early stopping | Thêm field vào `training/config.py` và dùng trong `training/trainer.py` |

### Ví dụ thêm layer vào classification head (`model/modeling.py`)

```python
self.classifier = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim),   # layer 1
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 256),          # layer 2 — thêm mới
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(256, num_labels),          # output
)
```

---

## Dataset

**[taidng/UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)** — Vietnamese Question Answering dataset.

| Split | Số mẫu |
|---|---|
| train | ~27 000 |
| validation | ~3 000 |

Dataset tự động tải về khi chạy lần đầu qua HuggingFace `datasets`.

---

## Output

Sau mỗi epoch, checkpoint được lưu tại:

```
outputs/checkpoints/
├── epoch-1/
│   ├── model.pt
│   ├── tokenizer_config.json
│   └── config.yaml        # config của run này
├── epoch-2/
├── epoch-3/
└── best/                  # checkpoint có valid_loss thấp nhất
    ├── model.pt
    ├── tokenizer_config.json
    └── config.yaml
```

