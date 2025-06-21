# utils/data_utils.py

import pickle
import json

def load_pickle_documents(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_claims(jsonl_path, exclude_nei=False):
    """
    Load claims from a jsonl file.

    Args:
        jsonl_path: Path to jsonl file.
        exclude_nei: If True, exclude claims with label 'NOT ENOUGH INFO'.

    Returns:
        List of claim dicts.
    """
    claims = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if exclude_nei:
                label = obj.get("label", "").strip().lower()
                if label == "not enough info":
                    continue
            claims.append(obj)
    return claims

# Example usage:
# claims = load_claims("data/test.jsonl", exclude_nei=True)