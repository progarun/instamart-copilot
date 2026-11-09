"""Part C: Support Ticket Triage with Guardrails"""
import pandas as pd
import re
import json
import random
import time
import mock_llm

def run_part_c():
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    segments = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")

    EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
    ORDER_RE = re.compile(r"ORD-\d{6}")

    SYSTEM_ZERO_SHOT = (
        "CLASSIFY the support ticket into one of: billing, delivery, technical, fraud.\n"
        'Return JSON only: {"category": "...", "reason": "..."}'
    )

    SYSTEM_FEW_SHOT = (
        "CLASSIFY the support ticket into one of: billing, delivery, technical, fraud.\n"
        'Return JSON only: {"category": "...", "reason": "..."}\n'
        "EXAMPLES:\n"
        'Ticket: Someone hacked my account and made unauthorized charges.\n{"category": "fraud", "reason": "unauthorized account access"}\n'
        'Ticket: The app crashed when I clicked checkout.\n{"category": "technical", "reason": "app stability failure"}'
    )

    def redact(text):
        t = EMAIL_RE.sub("[EMAIL]", text)
        t = PHONE_RE.sub("[PHONE]", t)
        t = ORDER_RE.sub("[ORDER_ID]", t)
        return t

    def call_with_retry(messages):
        for n in range(3):
            try:
                return mock_llm.chat(messages)
            except mock_llm.TransientLLMError:
                wait = 0.05 * (2 ** n) * random.uniform(0.9, 1.15)
                time.sleep(wait)
        return None

    # 1. Budget Demo Run (zero-shot, budget 6000)
    mock_llm.reset()
    running_tokens = 0
    budget_results = []
    for _, row in tickets.iterrows():
        tid, cid, txt = row["ticket_id"], row["customer_id"], row["text"]
        if not txt or not txt.strip():
            budget_results.append({"ticket_id": tid, "status": "skipped_empty"})
            continue
        if mock_llm.est_tokens(txt) > 400:
            budget_results.append({"ticket_id": tid, "status": "flagged_long"})
            continue
        if running_tokens >= 6000:
            budget_results.append({"ticket_id": tid, "status": "budget_exhausted"})
            continue

        resp = call_with_retry([{"role": "system", "content": SYSTEM_ZERO_SHOT}, {"role": "user", "content": redact(txt)}])
        running_tokens += resp.usage.total_tokens
        budget_results.append({"ticket_id": tid, "status": "classified"})

    pd.DataFrame(budget_results).to_csv("outputs/triage_budget_demo.csv", index=False)

    # 2. Final Triage Run (few-shot, budget 100000)
    mock_llm.reset()
    results = []
    for _, row in tickets.iterrows():
        tid, cid, txt = row["ticket_id"], row["customer_id"], row["text"]
        is_prem = bool(segments.loc[cid, "is_premium"]) if cid in segments.index else False
        sname = segments.loc[cid, "segment_name"] if cid in segments.index else ""

        if not txt or not txt.strip():
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "segment_name": sname, "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
        if mock_llm.est_tokens(txt) > 400:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "segment_name": sname, "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue

        redacted = redact(txt)
        resp = call_with_retry([{"role": "system", "content": SYSTEM_FEW_SHOT}, {"role": "user", "content": redacted}])
        toks = resp.usage.total_tokens
        content = resp.choices[0].message.content

        cat = None
        status = "flagged_invalid_output"
        try:
            data = json.loads(content)
            if data.get("category") in ["billing", "delivery", "technical", "fraud"]:
                status = "classified_json"
                cat = data["category"]
        except Exception:
            words = re.findall(r"[a-z]+", content.lower())
            found = [c for c in ["billing", "delivery", "technical", "fraud"] if c in words]
            if len(found) == 1:
                status = "classified_keyword"
                cat = found[0]

        if status == "flagged_invalid_output":
            route, priority = "human_review", "NORMAL"
        elif cat == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_prem:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"

        results.append({
            "ticket_id": tid, "customer_id": cid, "status": status,
            "category": cat or "", "segment_name": sname, "priority": priority, "route": route, "tokens_used": toks
        })

    pd.DataFrame(results).to_csv("outputs/triage_results.csv", index=False)
    print("Part C completed successfully.")

if __name__ == "__main__":
    run_part_c()
