"""Usage: python -m scripts.ingest data/jobs.jsonl [--partial]"""
import argparse
import time

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from aeromatch.config import settings
from aeromatch.db import init_schema
from aeromatch.ingest import ingest, load_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--partial", action="store_true", help="feed is a delta, don't deactivate missing jobs")
    a = p.parse_args()
    init_schema()
    jobs = load_jsonl(a.path)
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        t = time.perf_counter()
        stats = ingest(conn, jobs, full_snapshot=not a.partial)
        print(f"{stats} in {time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    main()
