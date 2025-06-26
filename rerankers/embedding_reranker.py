# rerankers/embedding_reranker.py
# qwen3 text embedding
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Reranker:
    def __init__(
        self,
        model_name="Qwen/Qwen3-Reranker-0.6B",
        device=None,
        instruction=None,
        batch_size=8,
        max_length=768,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = (
            instruction
            or "Given a statement, retrieve relevant passages that relate to the statement."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        )

        # Optionally: flash_attention_2 and fp16 (if your environment supports)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(self.device).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = '<|im_start|>system\nJudge whether the passage meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )

    def format_instruction(self, query, doc):
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
        tokenizer = self.tokenizer
        # Truncate each pair if needed to fit max_length
        inputs = tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        # Add prefix/suffix tokens
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        # Pad and move to device
        inputs = tokenizer.pad(
            inputs,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_length,
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs):
        model = self.model
        # Forward pass: logits on last token
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        # [no, yes] softmax for binary
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

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

    def rerank(self, claim, candidate_docs, topk=5):
        # claim: str; candidate_docs: list of {"doc_id", "score", "text"}
        pairs = [self.format_instruction(claim, doc["text"]) for doc in candidate_docs]
        # Batched
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            inputs = self.process_inputs(batch_pairs)
            batch_scores = self.compute_logits(inputs)
            scores.extend(batch_scores)
        # Attach scores and rerank
        reranked = []
        for doc, score in zip(candidate_docs, scores):
            doc = dict(doc)
            doc["qwen_score"] = float(score)
            reranked.append(doc)
        reranked = sorted(reranked, key=lambda x: x["qwen_score"], reverse=True)
        return reranked[:topk]


if __name__ == "__main__":
    reranker = Qwen3Reranker(
        model_name="Qwen/Qwen3-Reranker-0.6B",
        device=None,
        instruction=None,
        batch_size=8,
        max_length=768,
    )
    claim = "一行出家衆出家前的名字爲張遂."
    candidates = [
        {
            "text": "是古代希臘的哲學體系與學派之總稱的古希臘哲學的代表學派只有斯多葛學派"
        },
        {"text": "威廉·華茲華斯是生活在英格蘭湖區的民意代表。"},
    ]
    result = reranker.rerank(claim, candidates)
    print(result)


# from typing import Any

# from langchain_core.documents import Document
# # langchain wrapper
# from langchain_core.runnables import Runnable
# from pydantic import Field


# class LC_Reranker(Runnable):
#     custom_reranker: Any = Field(...)
#     took: int = Field(default=5)

#     def __init__(self, custom_reranker, took=5):
#         self.custom_reranker = custom_reranker
#         self.took = took

#     def invoke(self, input, **kwargs):
#         # input: {'query': str, 'docs': List[Document]}
#         query = input["query"]
#         docs = input["docs"]
#         candidates = [
#             {
#                 "doc_id": doc.metadata["doc_id"],
#                 "score": doc.metadata.get("bm25_score", 0),
#                 "text": doc.page_content,
#             }
#             for doc in docs
#         ]
#         reranked = self.custom_reranker.rerank(query, candidates, took=self.took)
#         return [Document(page_content=doc["text"], metadata=doc) for doc in reranked]
