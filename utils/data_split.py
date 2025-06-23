import os
import random

from preprocess_wiki import text_preprocess


def split_jsonl(input_path, train_path, test_path, train_ratio=0.8, seed=42):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.seed(seed)
    random.shuffle(lines)
    n_train = int(len(lines) * train_ratio)
    train_lines = lines[:n_train]
    test_lines = lines[n_train:]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(text_preprocess(train_lines))
    with open(test_path, "w", encoding="utf-8") as f:
        f.writelines(text_preprocess(test_lines))

    print(
        f"Total: {len(lines)}  |  Train: {len(train_lines)}  |  Test: {len(test_lines)}"
    )
    print(f"Train set: {train_path}")
    print(f"Test set: {test_path}")


# Usage example:
split_jsonl(
    input_path="../data/public_train.jsonl",
    train_path="../data/formated_train.jsonl",
    test_path="../data/formated_test.jsonl",
    train_ratio=0.8,
)
print("Split Done")
