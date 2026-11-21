"""Hybrid job retrieval: pgvector ANN + Postgres full-text, fused with Reciprocal Rank Fusion,
then re-ranked with the user's stored preferences and feedback."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .embeddings import get_embedder

CITY_ALIASES = {
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "blr": "Bengaluru",
    "gurgaon": "Gurugram", "gurugram": "Gurugram", "bombay": "Mumbai", "mumbai": "Mumbai",
    "hyderabad": "Hyderabad", "pune": "Pune", "chennai": "Chennai", "madras": "Chennai",
    "noida": "Noida", "delhi": "Delhi", "new delhi": "Delhi", "kolkata": "Kolkata",
}
RRF_K = 60
CANDIDATES = 50
SKILL_BOOST = 0.01  # one matching skill ~ moving up a few ranks in RRF space
# Weight of the keyword ranking relative to the vector ranking in RRF, tuned on the dev half of the
# eval set (scripts/eval_retrieval.py). Filtered queries already have a narrow candidate pool, where
# keyword noise hurts more, so they get a lower weight.
LEXICAL_WEIGHT = {"unfiltered": 0.75, "filtered": 0.25}


def normalise_city(city: str | None) -> str | None:
    if not city:
        return None
    return CITY_ALIASES.get(city.strip().lower(), city.strip().title())


class SearchFilters(BaseModel):
    city: str | None = Field(None, description="City name, e.g. Bengaluru, Pune")
    work_mode: Literal["onsite", "hybrid", "remote"] | None = None
    years_experience: int | None = Field(None, ge=0, le=40,
                                         description="Candidate's years of experience")
    min_salary_lpa: float | None = Field(None, ge=0, description="Minimum salary, lakh per annum")

    @field_validator("city")
    @classmethod
    def _city(cls, v: str | None) -> str | None:
        return normalise_city(v)


def _where(f: SearchFilters) -> tuple[str, list]:
    clauses, params = ["is_active"], []
    if f.city:
        clauses.append("city = %s")
        params.append(f.city)
    if f.work_mode:
        clauses.append("work_mode = %s")
        params.append(f.work_mode)
    if f.years_experience is not None:
        clauses.append("exp_min <= %s")
        params.append(f.years_experience)
    if f.min_salary_lpa is not None:
        clauses.append("salary_max_lpa >= %s")
        params.append(f.min_salary_lpa)
    return " AND ".join(clauses), params


def lexical_ids(conn, query: str, f: SearchFilters, k: int = CANDIDATES) -> list[int]:
    where, params = _where(f)
    # OR the query terms so long natural-language queries still match partially.
    sql = f"""
        WITH q AS (SELECT NULLIF(replace(plainto_tsquery('english', %s)::text, '&', '|'), '')::tsquery AS q)
        SELECT id FROM jobs, q
        WHERE {where} AND q.q IS NOT NULL AND tsv @@ q.q
        ORDER BY ts_rank_cd(tsv, q.q) DESC, id
        LIMIT %s"""
    return [r["id"] for r in conn.execute(sql, [query, *params, k]).fetchall()]


def vector_ids(conn, query: str, f: SearchFilters, k: int = CANDIDATES) -> list[int]:
    where, params = _where(f)
    qvec = get_embedder().embed([query])[0]
    if params:
        # HNSW applies WHERE clauses *after* the ANN walk, so a filter that keeps 25% of rows silently
        # drops most relevant hits from the ef_search window. Pre-filter, then exact KNN on the subset.
        sql = f"""WITH cand AS MATERIALIZED (SELECT id, embedding FROM jobs WHERE {where})
                  SELECT id FROM cand ORDER BY embedding <=> %s, id LIMIT %s"""
    else:
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        sql = f"SELECT id FROM jobs WHERE {where} ORDER BY embedding <=> %s LIMIT %s"
    return [r["id"] for r in conn.execute(sql, [*params, qvec, k]).fetchall()]


def rrf(*rankings: list[int], weights: tuple[float, ...] | None = None, k: int = RRF_K) -> dict[int, float]:
    weights = weights or (1.0,) * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank + 1)
    return scores


def _user_context(conn, user_id: int | None) -> tuple[set[str], set[int]]:
    if user_id is None:
        return set(), set()
    pref = conn.execute("SELECT skills FROM user_preferences WHERE user_id = %s", [user_id]).fetchone()
    dismissed = conn.execute(
        "SELECT job_id FROM user_job_actions WHERE user_id = %s AND action = 'dismissed'", [user_id]
    ).fetchall()
    return {s.lower() for s in (pref["skills"] if pref else [])}, {r["job_id"] for r in dismissed}


def search(conn, query: str, filters: SearchFilters | None = None, limit: int = 10,
           user_id: int | None = None,
           mode: Literal["hybrid", "vector", "lexical"] = "hybrid",
           lexical_weight: float | None = None) -> list[dict]:
    f = filters or SearchFilters()
    lex = lexical_ids(conn, query, f) if mode in ("hybrid", "lexical") else []
    vec = vector_ids(conn, query, f) if mode in ("hybrid", "vector") else []
    if lexical_weight is None:
        lexical_weight = LEXICAL_WEIGHT["filtered" if _where(f)[1] else "unfiltered"]
    scores = rrf(lex, vec, weights=(lexical_weight, 1.0))
    if not scores:
        return []

    user_skills, dismissed = _user_context(conn, user_id)
    rows = conn.execute(
        """SELECT j.id, j.title, c.name AS company, j.city, j.work_mode, j.exp_min, j.exp_max,
                  j.salary_min_lpa, j.salary_max_lpa, j.skills
           FROM jobs j JOIN companies c ON c.id = j.company_id WHERE j.id = ANY(%s)""",
        [list(scores)],
    ).fetchall()

    ranked = []
    for r in rows:
        if r["id"] in dismissed:
            continue
        overlap = len(user_skills & set(r["skills"]))
        r["score"] = round(scores[r["id"]] + SKILL_BOOST * overlap, 5)
        r["matched_skills"] = sorted(user_skills & set(r["skills"]))
        for key in ("salary_min_lpa", "salary_max_lpa"):
            r[key] = float(r[key]) if r[key] is not None else None
        ranked.append(r)
    ranked.sort(key=lambda r: (-r["score"], r["id"]))
    return ranked[:limit]
