from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings

_SCHEMA = Path(__file__).with_name("schema.sql")
_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def init_schema(database_url: str | None = None, dim: int | None = None) -> None:
    sql = _SCHEMA.read_text().replace("{dim}", str(dim or settings.embed_dim))
    with psycopg.connect(database_url or settings.database_url, autocommit=True) as conn:
        conn.execute(sql)


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(settings.database_url, min_size=1, max_size=10,
                               configure=_configure, kwargs={"row_factory": dict_row}, open=True)
    return _pool


@contextmanager
def connection():
    with pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
