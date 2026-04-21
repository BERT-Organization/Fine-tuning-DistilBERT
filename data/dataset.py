from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QAExample:
    """Một mẫu QA chuẩn SQuAD: question, context, answers."""

    question: str
    context: str
    answers: dict[str, list[Any]]
    id: str | None = None


def prepare_train_features(
    examples,
    tokenizer,
    question_column: str = "question",
    context_column: str = "context",
    answers_column: str = "answers",
    impossible_column: str = "is_impossible",
    max_length: int = 384,
    doc_stride: int = 128,
    padding: str = "max_length",
):
    """
    Chuyển raw QA examples thành tokenized features có start_positions/end_positions.

    Hỗ trợ kiểu dataset SQuAD/HuggingFace với schema:
        question: str
        context: str
        answers: {"text": [...], "answer_start": [...]}
    """
    pad_on_right = tokenizer.padding_side == "right"

    tokenized_examples = tokenizer(
        text=examples[question_column if pad_on_right else context_column],
        text_pair=examples[context_column if pad_on_right else question_column],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions: list[int] = []
    end_positions: list[int] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(batch_index=i)

        sample_index = sample_mapping[i]
        answers = examples[answers_column][sample_index]
        is_impossible = False
        if impossible_column in examples:
            is_impossible = bool(examples[impossible_column][sample_index])

        if is_impossible or len(answers.get("answer_start", [])) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples
