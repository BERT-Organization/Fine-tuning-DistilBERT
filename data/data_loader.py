from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict, load_dataset

from .dataset import prepare_train_features


def _infer_local_format(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    raise ValueError(f"Không hỗ trợ định dạng file: {file_path}")


def load_raw_datasets(config) -> DatasetDict:
    """Tải dataset QA từ HuggingFace Hub hoặc từ file local."""
    if config.dataset_name:
        return load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            cache_dir=config.cache_dir,
        )

    data_files: dict[str, str] = {}
    if config.train_file:
        data_files["train"] = config.train_file
    if config.validation_file:
        data_files["validation"] = config.validation_file
    if config.test_file:
        data_files["test"] = config.test_file

    if not data_files:
        raise ValueError("Cần cung cấp dataset_name hoặc ít nhất một trong train_file/validation_file/test_file.")

    data_format = _infer_local_format(next(iter(data_files.values())))
    load_kwargs = {"data_files": data_files, "cache_dir": config.cache_dir}
    if data_format == "csv" and next(iter(data_files.values())).endswith(".tsv"):
        load_kwargs["delimiter"] = "\t"

    return load_dataset(data_format, **load_kwargs)


def build_qa_datasets(tokenizer, config) -> DatasetDict:
    """Tokenize raw QA data và tạo start/end positions cho training."""
    raw_datasets = load_raw_datasets(config)

    processed = DatasetDict()
    for split_name in raw_datasets.keys():
        processed[split_name] = raw_datasets[split_name].map(
            lambda examples: prepare_train_features(
                examples,
                tokenizer=tokenizer,
                question_column=config.question_column,
                context_column=config.context_column,
                answers_column=config.answers_column,
                impossible_column=config.impossible_column,
                max_length=config.max_length,
                doc_stride=config.doc_stride,
                padding=config.padding,
            ),
            batched=True,
            remove_columns=raw_datasets[split_name].column_names,
        )
        processed[split_name].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
        )

    return processed
