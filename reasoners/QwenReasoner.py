# reasoner/reasoner_qwen.py

import gc

import torch
from peft import PeftConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from reasoners.prompting import build_claim_verification_prompt


class QwenReasoner:
    def __init__(
        self,
        model_name="Qwen/Qwen3-8B",
        model_path=None,
        device="auto",
        language="en",  # "en" or "zh"
        with_evidence=True,  # True or False
        max_new_tokens=512,
        thinking=False,
        output_w_reason=False,
        exclude_nei=False,
        use_int4=True,
    ):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.language = language
        self.with_evidence = with_evidence
        self.thinking = thinking
        self.output_w_reason = output_w_reason
        self.exclude_nei = exclude_nei
        # Int4 loading
        if use_int4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )
        else:
            bnb_config = None

        # Detect if using a PEFT/LoRA model
        try:
            peft_config = PeftConfig.from_pretrained(model_path)
            # Load base model then apply PEFT adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map=device,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
            print("using int4 model with Bnb")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        except Exception as e:
            print(e if int4 else "")
            print(f"using default {model_name} model")
            # Not a PEFT/LoRA model, just load as causal LM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )

    def build_prompt(self, claim, evidence_list):
        # Use centralized prompting logic
        return build_claim_verification_prompt(
            claim=claim,
            evidence_list=evidence_list,
            language=self.language,
            with_evidence=self.with_evidence,
            exclude_nei=self.exclude_nei,
            output_w_reason=self.output_w_reason,
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
        # TODO use recommended params
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
        model_name="Qwen/Qwen3-8B",  # ./models/qwen3_8b_reasoner_labelonly_ckpt Qwen/Qwen3-8B
        model_path="./models/qwen3_8b_reasoner_labelonly_ckpt/checkpoint-190",
        device="auto",
        language="zh",
        with_evidence=True,
        max_new_tokens=512,
        use_int4=True,
        output_w_reason=False,
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
