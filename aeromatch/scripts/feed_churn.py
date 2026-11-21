"""Simulate a day of feed churn (edits, removals, new postings) and measure incremental re-indexing
against a full re-embed. Usage: python -m scripts.feed_churn"""
import random
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from aeromatch.config import settings
from aeromatch.embeddings import get_embedder, job_text
from aeromatch.ingest import ingest, load_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"


def main(edit=0.05, remove=0.02, add=0.03) -> None:
    random.seed(7)
    jobs = load_jsonl(DATA / "jobs.jsonl")
    n = len(jobs)
    for j in random.sample(jobs, int(edit * n)):
        j.salary_max_lpa = round((j.salary_max_lpa or 10) * 1.1, 1)
    kept = random.sample(jobs, n - int(remove * n))
    new = [j.model_copy(update={"external_id": f"N{i:05d}"}) for i, j in enumerate(random.sample(jobs, int(add * n)))]
    feed = kept + new

    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        t = time.perf_counter()
        stats = ingest(conn, feed)
        inc = time.perf_counter() - t
        t = time.perf_counter()
        get_embedder().embed([job_text(j.title, j.skills, j.description) for j in feed])
        full_embed = time.perf_counter() - t
        print(f"{stats}\nincremental ingest {inc:.2f}s, re-embedded {stats.embedded}/{len(feed)} "
              f"({stats.embedded / len(feed):.0%}); embedding the full feed alone takes {full_embed:.2f}s")
        ingest(conn, load_jsonl(DATA / "jobs.jsonl"))  # restore the original feed


if __name__ == "__main__":
    main()
