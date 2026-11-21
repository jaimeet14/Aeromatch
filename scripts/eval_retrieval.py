"""Offline retrieval eval: lexical vs vector vs hybrid on the labelled query set.

Relevance: a job is relevant if it belongs to the query's role family (and, in the filtered
setting, satisfies the query's filters). Queries are split dev (even qid) / test (odd qid):
the hybrid lexical weight is tuned on dev and every number reported is on test.

Usage: python -m scripts.eval_retrieval"""
import json
import math
import statistics
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from aeromatch.config import settings
from aeromatch.search import SearchFilters, lexical_ids, rrf, vector_ids

DATA = Path(__file__).resolve().parent.parent / "data"
K = 10
GRID = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]


def metrics(ranked: list[int], relevant: set[int]) -> dict:
    gains = [1.0 if d in relevant else 0.0 for d in ranked[:K]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(K, len(relevant))))
    first = next((i for i, d in enumerate(ranked) if d in relevant), None)
    return {"ndcg@10": dcg / idcg, "recall@10": sum(gains) / min(K, len(relevant)),
            "mrr": 1 / (first + 1) if first is not None else 0.0, "p@5": sum(gains[:5]) / 5}


def fuse(lex: list[int], vec: list[int], w: float) -> list[int]:
    s = rrf(lex, vec, weights=(w, 1.0))
    return sorted(s, key=lambda d: (-s[d], d))[:K]


def mean(ms: list[dict]) -> dict:
    return {k: round(statistics.mean(m[k] for m in ms), 3) for k in ms[0]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ndcg", type=float, help="fail if tuned hybrid test nDCG@10 drops below this")
    args = ap.parse_args()
    queries = [json.loads(l) for l in open(DATA / "eval_queries.jsonl")]
    families = json.load(open(DATA / "job_families.json"))
    out: dict = {}
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        jobs = conn.execute("SELECT id, external_id, city, work_mode, exp_min FROM jobs WHERE is_active").fetchall()

        all_cases: dict[str, list] = {}
        for setting in ("no_filters", "with_filters"):
            cases = []  # (query, relevant, lexical ids, vector ids, lex ms, vec ms)
            for q in queries:
                f = SearchFilters(**q["filters"]) if setting == "with_filters" else SearchFilters()
                rel = {j["id"] for j in jobs if families[j["external_id"]] == q["family"]
                       and (not f.city or j["city"] == f.city)
                       and (not f.work_mode or j["work_mode"] == f.work_mode)
                       and (f.years_experience is None or j["exp_min"] <= f.years_experience)}
                if not rel:
                    continue
                t0 = time.perf_counter()
                lex = lexical_ids(conn, q["text"], f)
                t1 = time.perf_counter()
                with conn.transaction():
                    vec = vector_ids(conn, q["text"], f)
                t2 = time.perf_counter()
                cases.append((q, rel, lex, vec, (t1 - t0) * 1000, (t2 - t1) * 1000))
            all_cases[setting] = cases

    # One weight per regime (unfiltered / filtered), each tuned on that regime's dev queries only.
    weights = {}
    for setting, cases in all_cases.items():
        dev = [c for c in cases if c[0]["qid"] % 2 == 0]
        grid = {w: mean([metrics(fuse(c[2], c[3], w), c[1]) for c in dev])["ndcg@10"] for w in GRID}
        weights[setting] = max(grid, key=lambda w: (grid[w], -w))
        out[f"{setting}_dev_grid"] = grid
        print(f"[{setting}] dev nDCG@10 by lexical weight: {grid} -> w={weights[setting]}")
    out["lexical_weight"] = weights

    for setting, cases in all_cases.items():
        test = [c for c in cases if c[0]["qid"] % 2 == 1]
        systems = {"lexical": lambda c: c[2][:K], "vector": lambda c: c[3][:K],
                   "hybrid_rrf_equal": lambda c: fuse(c[2], c[3], 1.0),
                   "hybrid_rrf_tuned": lambda c, w=weights[setting]: fuse(c[2], c[3], w)}
        out[setting] = {"n_test": len(test), "systems": {}}
        print(f"\n[{setting}] test queries: {len(test)}")
        for name, rank in systems.items():
            rows = {"all": mean([metrics(rank(c), c[1]) for c in test])}
            for kind in ("title", "skills", "paraphrase"):
                rows[kind] = mean([metrics(rank(c), c[1]) for c in test if c[0]["kind"] == kind])
            out[setting]["systems"][name] = rows
            a = rows["all"]
            print(f"  {name:17s} nDCG@10={a['ndcg@10']:.3f} R@10={a['recall@10']:.3f} "
                  f"MRR={a['mrr']:.3f} P@5={a['p@5']:.3f} | "
                  + " ".join(f"{k}={rows[k]['ndcg@10']:.3f}" for k in ("title", "skills", "paraphrase")))
        lat = sorted(c[4] + c[5] for c in cases)
        out[setting]["hybrid_latency_ms"] = {"p50": round(statistics.median(lat), 1),
                                             "p95": round(lat[int(0.95 * len(lat)) - 1], 1)}
        print(f"  hybrid retrieval latency: {out[setting]['hybrid_latency_ms']}")

    (DATA.parent / "results").mkdir(exist_ok=True)
    json.dump(out, open(DATA.parent / "results" / "retrieval_eval.json", "w"), indent=2)
    if args.min_ndcg is not None:
        worst = min(out[s]["systems"]["hybrid_rrf_tuned"]["all"]["ndcg@10"] for s in all_cases)
        if worst < args.min_ndcg:
            raise SystemExit(f"retrieval regression: hybrid nDCG@10 {worst} < {args.min_ndcg}")
        print(f"quality gate passed: hybrid nDCG@10 >= {args.min_ndcg}")


if __name__ == "__main__":
    main()
