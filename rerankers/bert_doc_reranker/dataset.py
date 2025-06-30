# rerankers/bert_doc_reranker/dataset.py

import torch
from torch.utils.data import Dataset


class BertDocRerankerDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        """
        examples: list of dicts: {'query': str, 'doc': str, 'label': int}
        tokenizer: HuggingFace tokenizer
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query = ex["query"]
        doc = ex["doc"]
        label = ex["label"]
        encoding = self.tokenizer(
            query,
            doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
