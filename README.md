# Fine-tuning DistilBERT

Cau truc co ban cho du an fine-tuning text classification voi DistilBERT.

## Cau truc

- `data/`: dataset va loader
- `model/`: dinh nghia model
- `training/`: vong train / evaluation
- `scripts/`: entrypoint chay train
- `config/`: cau hinh mac dinh
- `outputs/`: checkpoint va log

## Dau vao du lieu

Du lieu demo duoc de nghia theo dinh dang TSV:

- cot 1: label so
- cot 2: van ban

Vi du:

```text
1\tThis movie was great
0\tI did not like this product
```

## Chay train

1. Cai dat dependencies:

```bash
pip install -r requirements.txt
```

2. Chuan bi `data/train.tsv` va `data/valid.tsv`.

3. Chay:

```bash
python scripts/train.py
```
