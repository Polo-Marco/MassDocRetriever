# rerankers/bert_doc_reranker/train.py

import json

from transformers import AutoModelForSequenceClassification, BertTokenizer

from rerankers.bert_doc_reranker.dataset import BertDocRerankerDataset
from rerankers.bert_doc_reranker.trainer import BertDocTrainer


def load_examples(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)  # List[{"query":..., "doc":..., "label":...}]


def main():
    # Load data
    train_examples = load_examples("data/doc_rerank_train.json")
    val_examples = load_examples("data/doc_rerank_dev.json")
    model_name = "hfl/chinese-pert-large"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = BertDocRerankerDataset(train_examples, tokenizer)
    val_dataset = BertDocRerankerDataset(val_examples, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    trainer = BertDocTrainer(
        model,
        train_dataset,
        val_dataset=val_dataset,
        batch_size=64,
        lr=2e-5,
        num_epochs=5,
        save_dir="models/test_lpert_ckpt",
        save_metric="recall",
    )
    trainer.train()


if __name__ == "__main__":
    main()
