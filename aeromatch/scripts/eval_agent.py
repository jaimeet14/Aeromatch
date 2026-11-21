"""End-to-end agent eval over scripted multi-turn conversations (data/agent_scenarios.json).

Per turn it checks: the expected tools were called, the filters that reached the search layer
(including ones recalled from memory) match, the reply came back as valid structured output on
the first try, and the recommended jobs belong to the right role family.

Usage:
  python -m scripts.eval_agent                       # LLM from LLM_BASE_URL / LLM_MODEL (e.g. Ollama)
  python -m scripts.eval_agent --model llama3.1:8b
  python -m scripts.eval_agent --baseline            # no LLM: raw message -> search -> top 3
"""
import argparse
import json
import statistics
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from aeromatch import memory
from aeromatch.agent import run_turn
from aeromatch.config import settings
from aeromatch.llm import LLMResponse, OpenAICompatibleLLM, ScriptedLLM, ToolCall

DATA = Path(__file__).resolve().parent.parent / "data"


def baseline_llm() -> ScriptedLLM:
    """Keyword-free control: search with the raw user message, recommend the top 3."""
    def step(msgs):
        if msgs[-1]["role"] == "user":
            return LLMResponse(None, [ToolCall("b1", "search_jobs", json.dumps({"query": msgs[-1]["content"]}))])
        res = json.loads(msgs[-1]["content"]).get("results", [])
        return LLMResponse(None, [ToolCall("b2", "final_answer", json.dumps(
            {"message": "Here are some matches.", "job_ids": [r["id"] for r in res[:3]]}))])
    return ScriptedLLM(step)


def applied_filters(trace: list[dict]) -> dict | None:
    for t in reversed(trace):
        if t["tool"] == "search_jobs" and t["ok"]:
            try:
                return json.loads(t["output"])["applied_filters"]
            except (json.JSONDecodeError, KeyError):
                return {}  # output preview was truncated; filters are at the start, so this is rare
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    scenarios = json.load(open(DATA / "agent_scenarios.json"))
    families = json.load(open(DATA / "job_families.json"))
    llm = None if a.baseline else OpenAICompatibleLLM(model=a.model)
    name = "baseline" if a.baseline else (a.model or settings.llm_model)

    rows = []
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        ext = {r["id"]: r["external_id"] for r in conn.execute("SELECT id, external_id FROM jobs").fetchall()}
        for sc in scenarios:
            uid = memory.create_user(conn, f"eval-{sc['id']}")
            cid = None
            for i, turn in enumerate(sc["turns"]):
                r = run_turn(conn, baseline_llm() if a.baseline else llm, uid, turn["user"], cid)
                cid = r.conversation_id
                called = [t["tool"] for t in r.trace if t["ok"]]
                row = {"scenario": sc["id"], "turn": i, "latency_ms": r.latency_ms, "steps": r.steps,
                       "structured": r.structured, "first_try": r.repairs == 0,
                       "tools_ok": all(t in called for t in turn["tools"]) and
                       (bool(turn["tools"]) or "search_jobs" not in called)}
                if "filters" in turn:
                    got = applied_filters(r.trace) or {}
                    row["filters_ok"] = all(got.get(k) == v for k, v in turn["filters"].items())
                if "family" in turn and r.answer.job_ids:
                    hits = [families.get(ext.get(j, ""), "") == turn["family"] for j in r.answer.job_ids]
                    row["precision"] = sum(hits) / len(hits)
                row["recommended"] = len(r.answer.job_ids)
                rows.append(row)
                print(f"{sc['id']:22s} t{i} tools={'OK ' if row['tools_ok'] else 'BAD'} "
                      f"filters={row.get('filters_ok', '-')!s:5s} prec={row.get('precision', '-')!s:5.5s} "
                      f"steps={r.steps} {r.latency_ms:.0f}ms  {called}")

    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    lat = sorted(r["latency_ms"] for r in rows)
    summary = {"model": name, "turns": len(rows), "tool_selection_acc": rate("tools_ok"),
               "filter_acc": rate("filters_ok"), "structured_rate": rate("structured"),
               "first_try_valid_rate": rate("first_try"), "recommendation_precision": rate("precision"),
               "mean_steps": round(statistics.mean(r["steps"] for r in rows), 2),
               "latency_ms_p50": round(statistics.median(lat)), "latency_ms_p95": round(lat[int(0.95 * len(lat)) - 1])}
    print(json.dumps(summary, indent=2))
    out = DATA.parent / "results"
    out.mkdir(exist_ok=True)
    json.dump({"summary": summary, "turns": rows},
              open(out / f"agent_eval_{name.replace(':', '_').replace('/', '_')}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
