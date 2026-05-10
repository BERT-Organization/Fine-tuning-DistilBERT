# Định dạng dữ liệu

Dự án nhận dữ liệu QA theo hướng SQuAD: mỗi mẫu gồm câu hỏi, context và danh sách đáp án trích trực tiếp từ context.

## 1. Dataset HuggingFace Hub

Mặc định:

```yaml
dataset_name: taidng/UIT-ViQuAD2.0
dataset_config_name: null
```

Dataset cần có tối thiểu các cột:

- `question`: câu hỏi.
- `context`: đoạn văn chứa đáp án.
- `answers`: dict chứa `text` và `answer_start`.
- `is_impossible`: tùy chọn, dùng cho câu hỏi không có đáp án.

Tên cột có thể đổi trong config:

```yaml
question_column: question
context_column: context
answers_column: answers
impossible_column: is_impossible
```

## 2. File local dạng đã flatten

Cách thuận tiện nhất là mỗi dòng/mỗi record tương ứng một mẫu QA.

Ví dụ JSONL:

```json
{"id":"q1","question":"Hà Nội là thủ đô của nước nào?","context":"Hà Nội là thủ đô của Việt Nam.","answers":{"text":["Việt Nam"],"answer_start":[21]},"is_impossible":false}
{"id":"q2","question":"Thành phố nào là thủ đô của Lào?","context":"Hà Nội là thủ đô của Việt Nam.","answers":{"text":[],"answer_start":[]},"is_impossible":true}
```

Ví dụ JSON dạng list:

```json
[
  {
    "id": "q1",
    "question": "Hà Nội là thủ đô của nước nào?",
    "context": "Hà Nội là thủ đô của Việt Nam.",
    "answers": {
      "text": ["Việt Nam"],
      "answer_start": [21]
    },
    "is_impossible": false
  }
]
```

Config:

```yaml
dataset_name: null
train_file: data/train.jsonl
validation_file: data/valid.jsonl
test_file: data/test.jsonl
```

## 3. CSV/TSV

CSV cần có cột `answers` là JSON string:

```csv
id,question,context,answers,is_impossible
q1,"Hà Nội là thủ đô của nước nào?","Hà Nội là thủ đô của Việt Nam.","{""text"": [""Việt Nam""], ""answer_start"": [21]}",false
```

TSV cũng được hỗ trợ qua extension `.tsv`.

## 4. SQuAD JSON lồng

Schema SQuAD gốc thường có dạng:

```json
{
  "version": "1.0",
  "data": [
    {
      "title": "Bài viết",
      "paragraphs": [
        {
          "context": "Hà Nội là thủ đô của Việt Nam.",
          "qas": [
            {
              "id": "q1",
              "question": "Hà Nội là thủ đô của nước nào?",
              "answers": [
                {
                  "text": "Việt Nam",
                  "answer_start": 21
                }
              ],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
```

Code hiện dùng `datasets.load_dataset("json")` cho file local, nên dạng flatten thường an toàn hơn. Nếu dữ liệu đang ở SQuAD JSON lồng, dùng script chính thức:

```bash
python scripts/flatten_squad.py \
  --input_file data/squad_nested.json \
  --output_file data/train_flatten.jsonl
```

## 5. Quy tắc cho `answers`

Mẫu có đáp án:

```json
{
  "answers": {
    "text": ["Việt Nam"],
    "answer_start": [21]
  },
  "is_impossible": false
}
```

Mẫu không có đáp án:

```json
{
  "answers": {
    "text": [],
    "answer_start": []
  },
  "is_impossible": true
}
```

`answer_start` là chỉ số ký tự 0-based trong context gốc. Chuỗi tại `context[answer_start:answer_start + len(text)]` phải khớp với đáp án.

Khi bật `use_vietnamese_segmentation`, pipeline sẽ segment context rồi tự căn chỉnh lại offset bằng `align_segmentation_offset` để map từ raw context sang segmented context. Dữ liệu đầu vào vẫn nên giữ `answer_start` theo context gốc.

## 6. Checklist trước khi train

- Câu hỏi và context không rỗng.
- File dùng UTF-8.
- `answers.text` và `answers.answer_start` là list.
- Với mẫu có đáp án, `answer_start` hợp lệ và khớp text trong context.
- Với mẫu không có đáp án, `answers.text` và `answers.answer_start` nên để list rỗng.
- Split train có dữ liệu.
- Split validation có `answers` nếu muốn tính validation loss và lưu `best_model`.
