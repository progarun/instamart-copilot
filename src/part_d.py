"""Part D: Policy RAG Assistant and Refund Agent"""
import os
import json
import time
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

# 1. Chunking
chunks = []
for fn in sorted(os.listdir("data/policies")):
    if fn.endswith(".txt"):
        words = open(os.path.join("data/policies", fn)).read().split()
        start = 0
        while start < len(words):
            chunk_words = words[start:start+40]
            chunks.append({"source": fn, "text": " ".join(chunk_words)})
            if start + 40 >= len(words):
                break
            start += 30

vec = TfidfVectorizer().fit([c["text"] for c in chunks])
chunk_tfidf = vec.transform([c["text"] for c in chunks])

def retrieve(query, k=3):
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, chunk_tfidf)[0]
    top_idx = sims.argsort()[::-1][:k]
    return [{"source": chunks[i]["source"], "score": float(sims[i]), "text": chunks[i]["text"]} for i in top_idx]

# 2. Tool implementations
def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    row = pd.read_sql_query(f"SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = '{order_id}'", conn)
    conn.close()
    if len(row) == 0:
        return {"error": "order not found"}
    r = row.iloc[0].to_dict()
    return {"order_id": r["order_id"], "status": r["status"], "order_amount": float(r["order_amount"]), "delivery_minutes": None if pd.isna(r["delivery_minutes"]) else int(r["delivery_minutes"])}

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}

def issue_refund(order_id, refund_inr):
    record = {"timestamp": time.time(), "tool": "issue_refund", "order_id": order_id, "refund_inr": float(refund_inr)}
    with open("outputs/refund_ledger.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    record = {"timestamp": time.time(), "tool": "escalate_to_human", "order_id": order_id, "refund_inr": float(refund_inr)}
    with open("outputs/refund_ledger.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": float(refund_inr)}

TOOL_MAP = {
    "get_order": get_order, "search_policy": search_policy,
    "calculate_refund": calculate_refund, "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id="run_1"):
    start_time = time.monotonic()
    messages = [{"role": "system", "content": "AGENT"}, {"role": "user", "content": user_message}]
    tokens_used = 0
    steps = []

    for iteration in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT", "final_answer": "Timeout reached; handoff to human.", "llm_calls": iteration, "tokens_used": tokens_used, "steps": steps}
        
        resp = mock_llm.chat(messages)
        tokens_used += resp.usage.total_tokens
        tc = resp.choices[0].message.tool_calls

        if tokens_used > token_limit:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Token limit exceeded; handoff to human.", "llm_calls": iteration + 1, "tokens_used": tokens_used, "steps": steps}

        if not tc:
            content = resp.choices[0].message.content
            status = content.split(":")[0].strip()
            return {"scenario_id": scenario_id, "status": status, "final_answer": content, "llm_calls": iteration + 1, "tokens_used": tokens_used, "steps": steps}

        call = tc[0]
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": call.id, "name": call.name, "arguments": call.arguments}]})
        obs = TOOL_MAP[call.name](**call.arguments)
        steps.append({"tool": call.name, "arguments": call.arguments, "observation": obs})
        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": json.dumps(obs)})

        with open(f"state/{run_id}.json", "w") as f:
            json.dump({"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "step_number": len(steps), "tools_executed": [s["tool"] for s in steps]}, f)

    return {"scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Max steps reached without resolution; handoff to human.", "llm_calls": max_iterations, "tokens_used": tokens_used, "steps": steps}

def run_part_d():
    # Clear previous ledgers
    if os.path.exists("outputs/refund_ledger.jsonl"):
        os.remove("outputs/refund_ledger.jsonl")
    if os.path.exists("outputs/agent_runs.jsonl"):
        os.remove("outputs/agent_runs.jsonl")

    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    for _, sc in scenarios.iterrows():
        res = run_agent(sc["message"], sc["scenario_id"], run_id=f"run_{sc['scenario_id']}")
        with open("outputs/agent_runs.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
    print("Part D completed successfully.")

if __name__ == "__main__":
    run_part_d()
