# prompting.py


def build_claim_verification_prompt(
    claim,
    evidence_list=None,
    language="en",
    with_evidence=True,
    exclude_nei=True,
    output_w_reason=False,
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
                f"[{i}] [{ev.get('doc_id', '')}] "
                if "doc_id" in ev or "line_id" in ev
                else ""
            )
            evidence_texts.append(f"{prefix}{ev['text']}")
        joined_evidence = "\n".join(evidence_texts)
    else:
        joined_evidence = ""

    if language == "en":
        if with_evidence:
            if exclude_nei:
                prompt = [
                    "You are an expert fact checker. ",
                    "Given the following claim and evidence, classify the claim as SUPPORTS or REFUTES. ",
                ]
                if output_w_reason:
                    prompt.append("Briefly explain your reason.\n\n")
                prompt += [
                    f"Claim: {claim}\n",
                    f"Evidence:\n{joined_evidence}\n",
                    "Output in the format:\n",
                    "label: <SUPPORTS/REFUTES>\n",
                ]  # exclude NEI for fair comparison
                if output_w_reason:
                    prompt.append("reason: <your explanation>\n")
                prompt = "\n".join(prompt)
            else:
                prompt = [
                    "You are an expert fact checker. ",
                    "Given the following claim and evidence, classify the claim as SUPPORTS, REFUTES or NOT ENOUGH INFO. ",
                ]
                if output_w_reason:
                    prompt.append("Briefly explain your reason.\n\n")
                prompt += [
                    f"Claim: {claim}\n",
                    f"Evidence:\n{joined_evidence}\n",
                    "Output in the format:\n",
                    "label: <SUPPORTS/REFUTES/NOT ENOUGH INFO>\n",
                ]  # exclude NEI for fair comparison
                if output_w_reason:
                    prompt.append("reason: <your explanation>\n")
                prompt = "\n".join(prompt)
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
            if exclude_nei:
                prompt = (
                    "你是一位推理專家。\n"
                    "根據下列論述與證據，請判斷該論述的真實性，分為 SUPPORTS（支持）、REFUTES（反駁），"
                    "並簡要說明你的判斷理由。\n\n"
                    f"論述：{claim}\n"
                    f"證據：\n{joined_evidence}\n"
                    "用以下格式回答：\n"
                    "label: <SUPPORTS/REFUTES>\n"
                    "reason: <你的理由>\n"
                )
            else:
                prompt = [
                    "你是一位推理專家。",
                    "根據下列論述與證據，請判斷該論述的真實性，分為 SUPPORTS（支持）、REFUTES（反駁）、NOT ENOUGH INFO（資訊不足）。",
                ]
                if output_w_reason:
                    prompt.append("並簡要說明你的判斷理由。")
                prompt += [
                    f"\n論述：{claim}",
                    f"證據：\n{joined_evidence}",
                    "用以下格式回答：",
                    "label: <SUPPORTS/REFUTES/NOT ENOUGH INFO>",
                ]
                if output_w_reason:
                    prompt.append("reason: <你的理由>")
                prompt = "\n".join(prompt)
        else:
            prompt = (
                "你是一位事實查核專家。\n"
                "根據你現有的知識，判斷該論述的真實性，分為 SUPPORTS（支持）、REFUTES（反駁），"
                "若你認為沒有足夠的信心做出判斷，標註 NOT ENOUGH INFO（資訊不足）。\n"
                "簡要說明你的判斷理由。\n\n"
                f"論述：{claim}\n"
                "用以下格式回答：\n"
                "label: <SUPPORTS/REFUTES/NOT ENOUGH INFO>\n"
                "reason: <你的理由>\n"
            )
    else:
        raise ValueError("Language must be 'en' or 'zh'.")
    return prompt


if __name__ == "__main__":
    claim = "天衛三軌道在天王星內部的磁層，以《仲夏夜之夢》作者緹坦妮雅命名。"
    candidate_evidence = [
        {"doc_id": "天衛三", "text": "天衛三是天王星的衛星之一。", "line_id": "39"},
        {
            "doc_id": "天衛三",
            "text": "該衛星的名字來自莎士比亞的作品。",
            "line_id": "41",
        },
    ]
    build_claim_verification_prompt(
        claim,
        evidence_list=candidate_evidence,
        language="zh",
        with_evidence=True,
        exclude_nei=False,
        output_w_reason=False,
    )
