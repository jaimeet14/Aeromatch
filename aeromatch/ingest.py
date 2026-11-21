"""Incremental indexer for a constantly changing job feed.

Each run takes a full snapshot of the feed, upserts changed postings, re-embeds only rows whose
content hash changed, and deactivates postings that disappeared from the feed."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .embeddings import get_embedder, job_text
from .search import normalise_city

BATCH = 512


class JobIn(BaseModel):
    external_id: str
    title: str
    company: str
    description: str
    skills: list[str] = Field(default_factory=list)
    city: str
    work_mode: str
    exp_min: int
    exp_max: int
    salary_min_lpa: float | None = None
    salary_max_lpa: float | None = None

    @field_validator("skills")
    @classmethod
    def _skills(cls, v: list[str]) -> list[str]:
        return sorted({s.strip().lower() for s in v if s.strip()})

    @field_validator("city")
    @classmethod
    def _city(cls, v: str) -> str:
        return normalise_city(v) or v

    def content_hash(self) -> str:
        payload = self.model_dump(exclude={"external_id"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class IngestStats:
    seen: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deactivated: int = 0
    reactivated: int = 0
    embedded: int = 0


def load_jsonl(path: str | Path) -> list[JobIn]:
    with open(path) as fh:
        return [JobIn.model_validate_json(line) for line in fh if line.strip()]


def ingest(conn, jobs: list[JobIn], full_snapshot: bool = True) -> IngestStats:
    stats = IngestStats(seen=len(jobs))
    existing = {r["external_id"]: (r["content_hash"], r["is_active"]) for r in
                conn.execute("SELECT external_id, content_hash, is_active FROM jobs").fetchall()}

    changed, revived = [], []
    for j in jobs:
        h = j.content_hash()
        old_hash, active = existing.get(j.external_id, (None, None))
        if old_hash == h:
            stats.unchanged += 1
            if not active:  # came back to the feed unchanged: no need to re-embed
                revived.append(j.external_id)
            continue
        changed.append((j, h))
        if j.external_id in existing:
            stats.updated += 1
        else:
            stats.inserted += 1

    company_ids: dict[str, int] = {}
    names = sorted({j.company for j, _ in changed})
    if names:
        conn.execute("INSERT INTO companies (name) SELECT unnest(%s::text[]) ON CONFLICT DO NOTHING", [names])
        company_ids = {r["name"]: r["id"] for r in conn.execute(
            "SELECT id, name FROM companies WHERE name = ANY(%s)", [names]).fetchall()}

    embedder = get_embedder()
    for start in range(0, len(changed), BATCH):
        batch = changed[start:start + BATCH]
        vecs = embedder.embed([job_text(j.title, j.skills, j.description) for j, _ in batch])
        stats.embedded += len(batch)
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO jobs (external_id, company_id, title, description, skills, city, work_mode,
                                     exp_min, exp_max, salary_min_lpa, salary_max_lpa, content_hash, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (external_id) DO UPDATE SET
                     company_id = EXCLUDED.company_id, title = EXCLUDED.title,
                     description = EXCLUDED.description, skills = EXCLUDED.skills, city = EXCLUDED.city,
                     work_mode = EXCLUDED.work_mode, exp_min = EXCLUDED.exp_min, exp_max = EXCLUDED.exp_max,
                     salary_min_lpa = EXCLUDED.salary_min_lpa, salary_max_lpa = EXCLUDED.salary_max_lpa,
                     content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding,
                     is_active = TRUE, updated_at = now()""",
                [(j.external_id, company_ids[j.company], j.title, j.description, j.skills, j.city,
                  j.work_mode, j.exp_min, j.exp_max, j.salary_min_lpa, j.salary_max_lpa, h, v)
                 for (j, h), v in zip(batch, vecs)],
            )

    if revived:
        stats.reactivated = conn.execute(
            "UPDATE jobs SET is_active = TRUE, updated_at = now() WHERE external_id = ANY(%s)", [revived]).rowcount

    if full_snapshot:
        live = [j.external_id for j in jobs]
        cur = conn.execute(
            "UPDATE jobs SET is_active = FALSE, updated_at = now() "
            "WHERE is_active AND NOT (external_id = ANY(%s))", [live])
        stats.deactivated = cur.rowcount
    conn.commit()
    return stats
