import gc

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, BertTokenizer


class BertDocReranker:
    def __init__(
        self,
        model_name="bert-base-chinese",
        batch_size=64,
        model_path="bert_doc_reranker_ckpt",
        device="cuda",
        debug=False,
        max_length=512,
    ):
        print(f"Loading model {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            device
        )
        # torch.cuda.empty_cache()#release pre-loaded gpu vram
        self.model.eval()
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size
        self.debug = debug

    def rerank(self, query, candidates, topk=10):
        device = self.device
        texts = [
            query + "[SEP]" + doc.get("doc_id", "") + ": " + doc["text"]
            for doc in candidates
        ]

        # Tokenize all at once for efficiency
        all_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        all_scores = []
        num_samples = len(texts)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        if self.debug:
            print(f"Batch size: {self.batch_size}")
            print(f"Total samples: {num_samples}")
            print(f"Num batches: {num_batches}")

        for i, start in enumerate(range(0, num_samples, self.batch_size)):
            batch_inputs = {
                k: v[start : start + self.batch_size].to(device)
                for k, v in all_inputs.items()
            }
            if self.debug:
                actual_bs = batch_inputs["input_ids"].shape[0]
                print(f"Batch {i+1}/{num_batches} - batch size: {actual_bs}")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / (1024**2)
                    reserved = torch.cuda.memory_reserved(device) / (1024**2)
                    print(
                        f"    CUDA mem allocated: {allocated:.2f}MB | reserved: {reserved:.2f}MB"
                    )
            with torch.no_grad():
                logits = self.model(**batch_inputs).logits
                batch_scores = softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_scores.extend(batch_scores)

        reranked = []
        for doc, score in zip(candidates, all_scores):
            doc = dict(doc)
            doc["score"] = float(score)
            reranked.append(doc)
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
        return reranked[:topk]

    def cleanup(self):
        """
        Safely release model (and tokenizer) from GPU/CPU memory.
        Call this when done with the instance.
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                if hasattr(self.model, "cpu"):
                    self.model.cpu()  # Move to CPU first (helps free GPU instantly)
                del self.model
                self.model = None
            if hasattr(self, "tokenizer"):
                del self.tokenizer
                self.tokenizer = None
        except Exception as e:
            print(f"[WARN] Exception during cleanup: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Instantiate your reranker
    reranker = BertDocReranker(model_path="models/test_bert_ckpt", device="cuda")

    # Suppose you have an example and a list of candidates
    example = {
        "claim": "太阳是银河系的中心",
        "evidence": [...],
        "label": "SUPPORTS",
    }
    candidates = [
        {"doc_id": "123", "text": "太阳位于银河系边缘的猎户臂上。"},
        {"doc_id": "456", "text": "银河系中心是一个超大质量黑洞。"},
        # ...
    ]

    result = reranker.rerank(example["claim"], candidates, topk=10)
    print(result)
