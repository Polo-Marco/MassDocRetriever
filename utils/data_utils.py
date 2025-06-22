# utils/data_utils.py

import pickle
import json
from utils.text_utils import text_preprocess
def load_pickle_documents(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def process_evidence(evidence, preprocess_func):
    # evidence is a list of lists of evidence spans
    new_evidence = []
    for group in evidence:
        new_group = []
        for span in group:
            # Copy to avoid modifying original
            span = list(span)
            # Preprocess the 3rd element (document ID) if it's a string
            if len(span) > 2 and isinstance(span[2], str):
                span[2] = preprocess_func(span[2])
            new_group.append(span)
        new_evidence.append(new_group)
    return new_evidence

def load_claims(jsonl_path, exclude_nei=False):
    """
    Load claims from a jsonl file, and apply preprocess_func to 'claim' and document IDs in 'evidence'.
    """
    claims = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            label = obj.get("label", "").strip().lower()
            if exclude_nei:
                if label == "not enough info":
                    continue
            # Preprocess the claim string
            if "claim" in obj:
                obj["claim"] = text_preprocess(obj["claim"])
            # Preprocess evidence document IDs
            if label != "not enough info" and "evidence" in obj:
                obj["evidence"] = process_evidence(obj["evidence"], text_preprocess)
            claims.append(obj)
    return claims

# Example usage:
if __name__ == "__main__":
    claims = load_claims("./data/test.jsonl",exclude_nei=False)
    print(claims[20:50])

