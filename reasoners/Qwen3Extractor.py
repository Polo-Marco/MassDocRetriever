import json
import re

from llama_cpp import Llama
from tqdm import tqdm

from utils.data_utils import load_jsonl_format


# Load your data
def load_examples(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# examples = load_examples("data/reason_extracted_train_2.json")
examples = load_jsonl_format("data/reason_extracted_train_5_HA_.jsonl")

# Setup Llama with thinking mode
llm = Llama(
    model_path="./models/qwen3-235/Qwen3-235B-A22B-Q4_K_M.gguf",
    n_gpu_layers=80,
    n_ctx=1024,
    use_mlock=False,
    verbose=False,
    enable_thinking=True,
)


def extract_reason_and_conclusion(text):
    # Match 理由: ... (non-greedy, up to 結論: or end)
    reason_pattern = r"理由[:：]\s*(.*?)(?:結論[:：]|$)"
    # Match 結論: ... (up to first line break or end of text)
    conclusion_pattern = r"結論[:：]\s*([^\n]*)"

    reason_match = re.search(reason_pattern, text, re.DOTALL)
    conclusion_match = re.search(conclusion_pattern, text)

    reason = reason_match.group(1).strip() if reason_match else ""
    conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

    return {"reason": reason, "conclusion": conclusion}


# Process each sample
results = []
for sample in tqdm(examples):
    label = sample["label"].upper()
    if (
        label != "NOT ENOUGH INFO"
        and sample["comment"] != ""
        and sample["conclusion"] == ""
    ):
        # second time distil with lower temperature
        # if len(sample["reason"])<10 or sample['conclusion']==None:
        # Format evidence as string
        evidence_str = "\n".join(
            f"{e['doc_id']}: {e['text']}" for e in sample["evidence"]
        )
        # Construct prompt
        prompt = f"Claim: {sample['claim']}\nEvidence:\n{evidence_str}\nPlease reason step by step."
        prompt = (
            "你是一位推理專家。\n"
            # "根據下列論述、完整證據與其標記為 <支持 / 反對 > 的結果，請用繁體中文整合所有證據進行合理推理，並說明為何這些證據共同導致該標記。\n"
            "根據下列論述、完整證據與其標記為 <支持 / 反對 > 的結果，請用繁體中文整合所有證據進行合理推理。\n"
            "請避免只使用部分證據，也不要省略推論鏈中的關鍵步驟。\n"
            f"論述： {sample['claim']}。\n"
            f"標記：{'支持' if sample['label'].upper()=='SUPPORTS' else '反對'}\n"
            "完整證據：\n"
            f"{evidence_str}\n"
            f"備註: {sample['comment']}"
            "請用以下格式回答：\n"
            "理由: <整合所有證據後，用繁體中文回答你的推理過程>\n"
            f"結論: <根據理由，用繁體中文回答導致{'支持' if label =='SUPPORTS' else '反對'}的原因>\n"
        )

        # LLM inference
        output = llm(
            prompt,
            max_tokens=512,
            stop=["</s>"],
            temperature=0.5,
            top_p=0.95,
            top_k=20,
            min_p=0,
            presence_penalty=1.7,
        )
        text = output["choices"][0]["text"]

        # Combine or keep as list
        # easoning = "\n".join(thinking_list) if thinking_list else text.strip()
        answers = extract_reason_and_conclusion(text)

        # Save with the original format, adding "reasoning"
        new_sample = dict(sample)
        new_sample["reason"] = answers["reason"]
        new_sample["conclusion"] = answers["conclusion"]
        new_sample["adjust"] = text
        print("===================================")
        print(f"Claim: {new_sample['claim']}")
        print(f"output: {text}")
        print(f"Reason: {new_sample['reason']}")
        print(f"Conclusion: {new_sample['conclusion']}")
        results.append(new_sample)
    else:
        results.append(sample)

# Save results
json.dump(results, open("data/reason_extracted_train_6_HA.json", "w"), indent=2)
print("Saved results with reasoning to data/reason_extracted_train_6_HA.json")
