"""
reasoner.Qwen3QaAgent.py
—————————
將 Claim-Verification 資料轉換成「必須使用所有證據才能回答」的多題 Q&A。
流程：Plan → Write → Review (迴圈)。
使用單一 llm() 函式（角色輪換）。
"""

import json
# ==========parse answers==============
import re
from typing import Any, Dict, List

from llama_cpp import Llama
from tqdm import tqdm

from utils.data_utils import load_jsonl_format

_Q_RE = re.compile(r"^\s*Question\s+(\d+)\s*:\s*(.*)", re.I)
_A_RE = re.compile(r"^\s*Answer\s+(\d+)\s*:\s*(.*)", re.I)

# 括號證據段   (證據: [0], [2] )
_EVID_RE = re.compile(r"\(\s*證據\s*:\s*([\s\S]*?)\)", re.I)
# 取 [  數字  ]，允許前後空白
_ID_RE = re.compile(r"\[\s*(\d+)\s*\]")


def _clean_text(txt: str) -> str:
    """去尾空白、保留內文換行"""
    return re.sub(r"[ \t]+\n", "\n", txt.strip())


def _split_answer_and_evidence(ans_text: str):
    """
    回傳 (answer_keep_evid_part, evidence_id_list)
    - 保留 (證據: …) 在 answer
    - 抽出整數 ID list
    """
    print(ans_text)
    m = _EVID_RE.search(ans_text)
    print(m)
    if not m:
        return _clean_text(ans_text), []
    evidence_ids = [int(x) for x in _ID_RE.findall(m.group(1))]
    return _clean_text(ans_text), evidence_ids


def parse_qa(plain_text: str) -> Dict[str, Any]:
    lines = plain_text.splitlines()
    qa_list: List[Dict[str, Any]] = []

    cur_question, cur_q_idx = [], None
    cur_answer, cur_a_idx = [], None

    def flush():
        if cur_q_idx is None:
            return
        ans = "\n".join(cur_answer)
        ans_clean, evid = _split_answer_and_evidence(ans)
        qa_list.append(
            dict(
                idx=cur_q_idx,
                question=_clean_text("\n".join(cur_question)),
                answer=ans_clean,
                evidence=evid,
            )
        )

    for ln in lines:
        if m := _Q_RE.match(ln):
            # 若已有上一題，先寫入
            if cur_question or cur_answer:
                flush()
                cur_question, cur_answer = [], []
            cur_q_idx = int(m.group(1))
            cur_question.append(m.group(2))
        elif m := _A_RE.match(ln):
            cur_a_idx = int(m.group(1))  # 不特別檢查一致性
            cur_answer.append(m.group(2))
        else:
            # 隨著區塊追加
            (cur_answer if cur_answer else cur_question).append(ln)

    # 收最後一題
    if cur_question or cur_answer:
        flush()

    return {"questions": qa_list}


from llama_cpp import Llama, LlamaTokenizer

# ======== llama.cpp 初始化 =========
# 量化好的 GGUF 模型路徑（自行替換）
MODEL_PATH = "./models/qwen3-235/Qwen3-235B-A22B-Q4_K_M.gguf"
# MODEL_PATH = "./models/qwen3-32/Qwen3-32B-Q6_K.gguf"

# 一次載入，後續多次呼叫會共用
llm_raw = Llama(  # ← 名稱隨意，只要跟下面 wrapper 對應
    model_path=MODEL_PATH,
    n_gpu_layers=60,  # 依顯卡 VRAM 而定
    n_ctx=4096,  # 建議拉到 4k~8k，1024 容易爆 context
    use_mlock=False,
    verbose=False,
    enable_thinking=True,
)
# 先加一個估算 tokenizer，最簡單直接用 tiktoken 或 llama_cpp 內建
tokenizer = LlamaTokenizer(llm_raw)


def llm(prompt: str, **kw) -> str:
    """
    給 two_step_self_eval 使用的薄包裝：
    - 直接呼叫你現成的 llm_raw(prompt, …)
    - 若回傳是 dict（新版 .create_completion 格式），則取 .['choices'][0]['text']
      若回傳是 str（舊版 __call__ overload），直接回傳
    """
    result = llm_raw(prompt, **kw)

    # 兼容不同 llama_cpp 版本
    if isinstance(result, dict):
        # print(result["choices"][0]["text"])
        return result["choices"][0]["text"]
    return result  # 已是字串


# ===================== 共享 LLM 參數 =====================


BASE_KWARGS = dict(
    max_tokens=512,
    stop=["</s>"],
    top_p=0.95,
    top_k=40,
    min_p=0,
    presence_penalty=1.2,
)

TEMP_PLAN = 0.3  # Step-1 規劃：低隨機
TEMP_WRITE = 0.6  # Step-2 撰寫：中隨機
TEMP_REVIEW = 0.0  # Step-2a 評審：Deterministic

# ===================== Prompt 範本 =====================

# ---------- 1. PLAN ----------
PLAN_TEMPLATE = """
你是一位 QA 規劃專家。
任務：依下列輸入，設計「問題藍圖」，使回答者必須閱讀所有證據才能回答所有題目；
- 每一組 QA 至少要引用 1 個證據，且**每組最多只能引用 2 個證據**。
- 整體題數不得超過 3 組。

輸入資訊
- label: {label}
- reason: {reason}
- conclusion: {conclusion}

請**只輸出**下方框線區塊（包含起訖標籤），不要額外解釋或加入其他文字。

===== PLAN START =====
- Q1: 使用證據 [0, ...]     # 每題最多填 2 個證據 ID
- Q2: 使用證據 [0]          # 如無需要可以省略
===== PLAN END =====

claim: {claim}
evidence: {evidence_block}
"""

# ---------- 2. WRITE ----------
WRITE_TEMPLATE = """
你是一位專業內容撰寫者。
**依照下方 PLAN 列出的題號與對應證據**，撰寫純文字 Q&A（不得加入 Markdown 標記）。
注意：`reason` 僅供你思考，請勿在輸出中洩漏。

===== PLAN =====
{plan}
===== END PLAN =====

證據（格式：[ID] : DocID: 內容）：
{evidence_block}

# internal_reason:
{reason}

撰寫規範：
1. 嚴格按照 PLAN 的 Q1 ~ Qn（最多 3 題）撰寫 Question 與 Answer，不可增刪題號，也不可替換證據。
2. **Question（問題）要求：**
   • 以自然、生活化的問法切入，問題要讓讀者有興趣或實際需要了解該資訊。
   • 不得出現任何證據標籤 `[index]`。
   • 不可直接引用證據句子或 claim（論述）原文。
   • 不得出現「根據/請指出 claim 或論述…」等字眼。
   • **請避免過於牽強、吹毛求疵、或「硬要拆分」論述內容的問句。**
   • 若證據本身資訊有限，可用泛問或描述式問題（如「請簡介…」「有哪些重點？」）。
   • 不必每題都對應論述一個片段，可依情境用自然問法整合多個資訊。
3. **Answer 內文** 應在適當位置插入對應的證據標籤，例如「……山林調查[0]……該次調查[1]……」。
4. 每題 Answer 結尾必附上 (證據: [0], [1])，且證據順序與 PLAN 保持一致。
5. 不可直接複製 claim 原句；全文使用繁體中文。

輸出格式（純文字範例）：
Question 1:
林務局首次大規模山林調查於何時展開？是誰主導該調查？

Answer 1:
林務局於 1879 年展開首次大規模山林調查[0]，由張山主要負責該次調查[1]。 (證據: [0], [1])

Question 2:
<...>

（僅列 PLAN 指定題號；嚴禁新增或遺漏題號）
"""


# ---------- REVIEW ----------
REVIEW_TEMPLATE = """
你是一位嚴謹的審稿員，請依下列規則逐項檢查 Q&A 是否合格（滿分 5 分）：

1. **題數與 PLAN 一致**：題號必須完全對應 PLAN（不得多於 3 題，也不得缺題或重複）。
2. **證據使用正確**
   2-a. 每題 Answer 內文與結尾 `(證據: …)` 均只出現 PLAN 指定的 ID，且順序相同。
   2-b. 所有 PLAN 中的證據 ID：{eids} 皆至少被引用一次。
   2-c. Question 內文不得包含任何 `[index]` 證據標籤。
3. **禁止提及 claim**
   - Question 內文不得出現「claim 論述」字樣，亦不得使用「根據/請指出 claim 或論述…」等措辭。
4. **格式**
   - 行首標籤必須為 `Question n:` / `Answer n:`。
   - Answer 內文需嵌入 `[index]` 標籤。
   - 每題 Answer 結尾必有 `(證據: […])`；Question 行尾不得有。
   - 全文不得含 Markdown 標記（如 `#`, `*`, `**`, `_` 等）。
5. **內容**
   - 不可直接複製 claim 原句。
   - 問句需自然通順，未見吹毛求疵或過於牽強的問法。
   - 全文使用繁體中文，敘述通順無重大語病。
6. **整體一致性**：無格式錯誤，邏輯清晰。

請輸出：
SCORE: <0~5>
FEEDBACK: <若 SCORE <5，簡述主要問題；若 SCORE=5，回覆 "OK">

Q&A：
{qa_block}
"""


def get_ctx_val(n_ctx_field):
    """如果是 callable 就呼叫；若本身是 int 直接回傳。"""
    return n_ctx_field() if callable(n_ctx_field) else n_ctx_field


def dynamic_max_tokens(n_ctx, prompt_tokens, reserve=64):
    """
    n_ctx       : 模型上下文長度 (e.g. 4096 或 8192)
    prompt_tokens: 送進模型的 prompt token 數
    reserve     : 留給安全區的 buffer，避免貼到極限
    回傳可用的 max_tokens
    """
    n_ctx_val = get_ctx_val(n_ctx)
    return max(64, min(1024, n_ctx_val - prompt_tokens - reserve))


def token_count(text: str) -> int:
    return len(tokenizer.encode(text))


# 修改 call_llm wrapper，動態塞 max_tokens
def call_llm(prompt: str, temperature: float, role: str, **extra):
    prompt_len = token_count(prompt)
    print("********PROMPT********")
    print(prompt)
    if role == "writer":
        extra.setdefault(
            "max_tokens", dynamic_max_tokens(llm_raw.n_ctx, prompt_len, 64)
        )
    else:  # planner & reviewer max_token 512
        extra.setdefault(
            "max_tokens", min(768, dynamic_max_tokens(llm_raw.n_ctx, prompt_len, 96))
        )
    kwargs = BASE_KWARGS | extra
    return llm(prompt, temperature=temperature, **kwargs)


# ===================== Planner =====================

import re
from typing import List

# -------- 取 PLAN 區塊 --------
# 允許： = PLAN START =、== PLAN START ==、===== PLAN START=====  …皆可
_PLAN_BLOCK_RE = re.compile(
    r"=+\s*PLAN\s+START\s*=+\s*(.*?)\s*=+\s*PLAN\s+END\s*=+",
    re.S | re.I,
)


def extract_plan(text: str) -> tuple[bool, str]:
    m = _PLAN_BLOCK_RE.search(text)
    if not m:
        return False, "找不到 PLAN 區塊，請務必以等號包圍 PLAN START/END。"
    plan_block = m.group(1).strip()
    if not plan_block:
        return False, "PLAN 區塊為空，請輸出正確格式。"
    return True, plan_block


# -------- 抓 ID --------
_ID_RE = re.compile(r"\[\s*([0-9, ]+)\s*\]")


def parse_ids(text: str) -> list[int]:
    """只抓 [] 內的所有數字，去重且保留順序，型別為 int"""
    ids = []
    for m in _ID_RE.finditer(text):
        for num in re.findall(r"\d+", m.group(1)):
            num = int(num)
            if num not in ids:
                ids.append(num)
    return ids


# -------- 核心檢查 --------
_LINE_RE = re.compile(r"^-\s*Q\d+\s*:\s*使用證據\s*\[([0-9,\s]+)\]\s*$")


def check_plan(plan_block: str, eids: List[int]) -> tuple[bool, str]:
    allowed = set([int(re.sub(r"[\[\]]", "", eid)) for eid in eids])
    used = set()
    n_lines = 0

    for line in plan_block.splitlines():
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        m = _LINE_RE.match(line)
        if not m:
            return False, f"格式錯誤：行「{line}」應為 - Qn: 使用證據 [0,1]"
        # 檢查證據
        ids = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if not ids:
            return False, f"每一組 QA 必須至少引用 1 個證據，錯誤行：{line}"
        if len(ids) > 2:
            return (
                False,
                f"每組 QA 最多只能引用 2 個證據（你的第 {n_lines} 行用了 {len(ids)} 個）。",
            )
        used.update(ids)

    if n_lines == 0:
        return False, "PLAN 內容為空。"
    # 捏造 or 遺漏
    fake = used - allowed
    if fake:
        return False, f"請不要捏造證據：出現了不存在的證據 ID {sorted(list(fake))}\n"
    missing = allowed - used
    if missing:
        return False, f"並未覆蓋所有證據：缺少 ID {sorted(list(missing))}\n"

    return True, ""


def get_plan(
    claim: str,
    eids: list[int],  # eids 建議直接傳 int
    evidence_block: str,
    label: str,
    reason: str,
    conclusion: str,
    max_retry: int = 3,
) -> str:
    """
    反覆呼叫 Planner → 檢查 Plan
    - 若區塊缺失、格式錯誤或證據不符，立即重試
    - 成功時回傳乾淨 Plan 文字
    """
    feedback = ""
    for attempt in range(1, max_retry + 1):
        print(f"========== Planner Call #{attempt} ==========")
        plan_prompt = PLAN_TEMPLATE.format(
            claim=claim,
            evidence_block=evidence_block,
            label=label,
            reason=reason,
            conclusion=conclusion,
        )
        if feedback:
            plan_prompt += f"\n\n# 上一次 reviewer 指出：{feedback} 請修正後重新規劃。"
        raw_plan = call_llm(
            plan_prompt,
            TEMP_PLAN,
            role="planner",
        )
        ok, plan_or_fb = extract_plan(raw_plan)
        if not ok:
            feedback = plan_or_fb
            print(feedback + "，重新生成…")
            continue

        extracted_plan = plan_or_fb
        print("extracted plan")
        print(extracted_plan)
        passed, fb2 = check_plan(extracted_plan, eids)
        if passed:
            return extracted_plan
        else:
            feedback = fb2
            print(feedback + "，重新生成…")
    raise RuntimeError("Planner 在多次嘗試後仍無法生成合格的 PLAN")


# ===================== Writer =====================


def write_qa(plan: str, evidence_block: str, reason: str, conclusion: str):
    print("==========writer===========")
    qas = call_llm(
        WRITE_TEMPLATE.format(
            plan=plan,
            evidence_block=evidence_block,
            reason=reason,
            conclusion=conclusion,
        ),
        TEMP_WRITE,
        role="writer",
    )
    print(qas)
    return qas


# ===================== Reviewer =====================


def review_qa(qa_block: str, eids: List[str], conclusion: str):
    print("==========reviewer===========")
    review = call_llm(
        REVIEW_TEMPLATE.format(qa_block=qa_block, eids=eids, conclusion=conclusion),
        TEMP_REVIEW,
        role="reviewer",
    )
    m = re.search(r"SCORE:\s*(\d+)", review)
    print(review)
    score = int(m.group(1)) if m else 0
    return score, review


# ===================== 主管線函式 =====================


def claim_to_qa(
    claim: str,
    evidence_dict: Dict[str, str],
    label: str,
    reason: str,
    conclusion: str,
    outer_try=3,
    inner_try=3,
) -> str:
    eids = list(evidence_dict.keys())
    evidence_block = "\n".join(f"{k}: {v}" for k, v in evidence_dict.items())

    # Step-1 計畫
    plan = get_plan(
        claim, eids, evidence_block, label, reason, conclusion, max_retry=outer_try
    )
    print("###### STRICT PLAN ######")
    print(plan)
    # Step-2 撰寫 + Reviewer Loop
    for _ in range(inner_try):
        qa = write_qa(plan, evidence_block, reason, conclusion)
        score, _ = review_qa(qa, eids, conclusion)
        if score >= 5:
            return qa.strip()
        # 若沒過：附 reviewer feedback 回寫 plan 讓模型調整
        plan += "\n# reviewer 指出需修正，上次未過檢查，請重新分配或增補證據。\n"
    raise RuntimeError("Writer 反覆嘗試後仍未通過 Reviewer 檢查")


# ===================== DEMO =====================
if __name__ == "__main__":
    data = load_jsonl_format("data/qa_extracted.jsonl")
    save_path = "data/qa_extracted_1.jsonl"
    for d in tqdm(data):  # good ex: 2712 14
        if d["label"].upper() == "NOT ENOUGH INFO":
            continue
        if d["qa"] == "失敗":
            claim = d["claim"]
            label = d["label"]
            evidence = {
                f"[{idx}]": f"[{i['doc_id']}] {i['text']}"
                for idx, i in enumerate(d["evidence"])
            }
            reason = d["reason"]
            conclusion = (
                d["conclusion"].split("。")[0]
                if "。" in d["conclusion"]
                else d["conclusion"]
            )
            print(claim, evidence, reason, conclusion)
            try:
                qa_result = claim_to_qa(claim, evidence, label, reason, conclusion)
                print("\n===== 生成結果 =====\n")
                print(qa_result)
                # parsed = parse_qa(qa_result)
                # print(parsed)
                # d['qa_parsed']=parsed
                d["qa"] = qa_result
            except RuntimeError as err:
                d["qa"] = "失敗"
                print("生成失敗：", err)
            with open(save_path, "a", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
                f.write("\n")
        else:
            with open(save_path, "a", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
                f.write("\n")
#     claim = "一行出家前名為張遂；他是唐代比丘且精通曆法。"
#     evidence = {
#         "E1": "《新唐書》記載：一行俗名張遂。",
#         "E2": "《舊唐書》載：張遂出家後法號一行，為唐代比丘。",
#         "E3": "史料說明：一行曾主持《大衍曆》編制並改進曆法。",
#     }
#     label       = "SUPPORTS"
#     reason      = "（略）完整的鏈式推理內容"
#     conclusion  = "一行確為張遂，且為唐代比丘與曆法專家。"

#     try:
#         qa_markdown = claim_to_qa(claim, evidence, label, reason, conclusion)
#         print("\n===== 生成結果 =====\n")
#         print(qa_markdown)
#     except RuntimeError as err:
#         print("生成失敗：", err)
