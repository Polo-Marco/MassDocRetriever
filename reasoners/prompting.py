# prompting.py


def build_claim_verification_prompt(
    claim, evidence_list=None, language="en", with_evidence=True
):
    """
    Build prompt for claim verification.
    - claim: str
    - evidence_list: list of dicts, each should have 'text', can have 'doc_id', 'line_id'
    - language: "en" or "zh"
    - with_evidence: bool
    Returns: prompt string
    """
    if evidence_list is None:
        evidence_list = []

    if with_evidence:
        evidence_texts = []
        for i, ev in enumerate(evidence_list):
            prefix = (
                f"[{ev.get('doc_id', '')}:{ev.get('line_id', '')}] "
                if "doc_id" in ev or "line_id" in ev
                else ""
            )
            evidence_texts.append(f"{prefix}{ev['text']}")
        joined_evidence = "\n".join(evidence_texts)
    else:
        joined_evidence = ""

    if language == "en":
        if with_evidence:
            prompt = (
                "You are an expert fact checker. "
                "Given the following claim and evidence, classify the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO, "
                "and briefly explain your reasoning.\n\n"
                f"Claim: {claim}\n"
                f"Evidence:\n{joined_evidence}\n"
                "Output in the format:\n"
                "label: <SUPPORTS/REFUTES>\n"  # exclude NEI for fair comparison
                "reason: <your explanation>\n"
            )
        else:
            prompt = (
                "You are an expert fact checker. "
                "Given the following claim, classify the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO, "
                "and briefly explain your reasoning.\n\n"
                f"Claim: {claim}\n"
                "Output in the format:\n"
                "label: <SUPPORTS/REFUTES/NOT ENOUGH INFO>\n"
                "reason: <your explanation>\n"
            )
    elif language == "zh":
        if with_evidence:
            prompt = (
                "你是一位資深事實查核專家。"
                "根據下列論述與證據，請判斷該論述的真實性，分為 SUPPORTS（支持）、REFUTES（反駁）、NOT ENOUGH INFO（資訊不足），"
                "並簡要說明你的判斷理由。\n\n"
                f"論述: {claim}\n"
                f"證據:\n{joined_evidence}\n"
                "請用以下格式回答：\n"
                "label: <SUPPORTS/REFUTES>\n"  # exclude NEI for fair comparison
                "reason: <你的說明>\n"
            )
        else:
            prompt = (
                "你是一位資深事實查核專家。"
                "根據下列論述，請判斷該論述的真實性，分為 SUPPORTS（支持）、REFUTES（反駁）、NOT ENOUGH INFO（資訊不足），"
                "並簡要說明你的判斷理由。\n\n"
                f"論述: {claim}\n"
                "請用以下格式回答：\n"
                "label: <SUPPORTS/REFUTES/NOT ENOUGH INFO>\n"
                "reason: <你的說明>\n"
            )
    else:
        raise ValueError("Language must be 'en' or 'zh'.")

    return prompt


# Optionally, you can add more functions for more prompt styles, etc.
