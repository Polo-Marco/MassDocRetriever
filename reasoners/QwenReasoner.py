# reasoner/reasoner_qwen.py

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoners.prompting import build_claim_verification_prompt


class QwenReasoner:
    def __init__(
        self,
        model_name="Qwen/Qwen3-8B",
        device="auto",
        language="en",  # "en" or "zh"
        with_evidence=True,  # True or False
        max_new_tokens=512,
        thinking=False,
        exclude_nei=False,
    ):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=device
        )
        self.language = language
        self.with_evidence = with_evidence
        self.thinking = thinking
        self.exclude_nei = exclude_nei

    def build_prompt(self, claim, evidence_list):
        # Use centralized prompting logic
        return build_claim_verification_prompt(
            claim=claim,
            evidence_list=evidence_list,
            language=self.language,
            with_evidence=self.with_evidence,
            exclude_nei=self.exclude_nei,
        )

    def reason(self, claim, evidence):
        prompt = self.build_prompt(claim, evidence)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,  # Reasoning doesn't need to be long
            temperature=0.2,
            num_beams=1,  # for efficiency
            do_sample=False,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Try to parse out the answer (with/without <think> blocks)
        try:
            index = len(output_ids) - output_ids[::-1].index(
                self.tokenizer.convert_tokens_to_ids("</think>")
            )
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        # Parse label/reason from content
        label, reason = self._parse_output(content)

        return {
            "label": label,
            "reason": reason,
            "raw_output": content,
            "thinking": thinking_content,
        }

    def reason_batch(self, batch_claims, batch_evidence_lists):
        """
        batch_claims: List[str]
        batch_evidence_lists: List[List[dict]]
        Returns: List[dict] (same as single reason)
        """
        # Build prompts for the batch
        prompts = [
            self.build_prompt(claim, evidence)
            for claim, evidence in zip(batch_claims, batch_evidence_lists)
        ]

        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
        text_batch = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
            for messages in messages_batch
        ]

        # Tokenize in batch (pad to longest in batch)
        model_inputs = self.tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        # Generate in batch
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            num_beams=1,
            do_sample=False,
        )

        # Parse output for each example
        outputs = []
        input_lens = [
            sum(x != self.tokenizer.pad_token_id for x in ids)
            for ids in model_inputs.input_ids
        ]
        for i, input_len in enumerate(input_lens):
            output_ids = generated_ids[i][input_len:].tolist()
            # Optional: parse <think> block
            try:
                idx = len(output_ids) - output_ids[::-1].index(
                    self.tokenizer.convert_tokens_to_ids("</think>")
                )
            except ValueError:
                idx = 0
            thinking_content = self.tokenizer.decode(
                output_ids[:idx], skip_special_tokens=True
            ).strip("\n")
            content = self.tokenizer.decode(
                output_ids[idx:], skip_special_tokens=True
            ).strip("\n")
            label, reason = self._parse_output(content)
            outputs.append(
                {
                    "label": label,
                    "reason": reason,
                    "raw_output": content,
                    "thinking": thinking_content,
                }
            )
        return outputs

    def _parse_output(self, text):
        """
        Attempts to extract 'label: ...' and 'reason: ...' from the output text.
        Falls back to raw string if not found.
        """
        import re

        label, reason = None, None
        # Robust matching
        m_label = re.search(
            r"label\s*[:：]\s*\**\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)\s*\**",
            text,
            re.I,
        )
        m_reason = re.search(r"reason\s*[:：]\s*([^\n]+)", text, re.I)
        if m_label:
            label = m_label.group(1).upper()
        if m_reason:
            reason = m_reason.group(1).strip()
        # Fallbacks
        if not label:
            label = "NOT ENOUGH INFO"
        if not reason:
            reason = text.strip()
        return label, reason

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
    reasoner = QwenReasoner(
        model_name="Qwen/Qwen3-8B",
        device="auto",
        language="zh",
        with_evidence=True,
        max_new_tokens=512,
    )
    claim = "天衛三軌道在天王星內部的磁層，以《仲夏夜之夢》作者緹坦妮雅命名。"
    candidate_evidence = [
        {"doc_id": "天衛三", "text": "天衛三是天王星的衛星之一。", "line_id": "39"},
        {
            "doc_id": "天衛三",
            "text": "該衛星的名字來自莎士比亞的作品。",
            "line_id": "41",
        },
    ]
    result = reasoner.reason(claim, candidate_evidence)
    print(result)
