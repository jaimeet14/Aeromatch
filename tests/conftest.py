import os

# Must be set before aeromatch.config is imported.
ADMIN_URL = os.getenv("TEST_ADMIN_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
TEST_DB = "aeromatch_test"
os.environ["DATABASE_URL"] = ADMIN_URL.rsplit("/", 1)[0] + "/" + TEST_DB
os.environ["EMBEDDER"] = "hash"
os.environ["EMBED_DIM"] = "256"

import psycopg  # noqa: E402
import pytest  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from aeromatch.db import init_schema  # noqa: E402
from aeromatch.ingest import JobIn, ingest  # noqa: E402

JOBS = [
    dict(external_id="A1", title="Backend Engineer (Python)", company="NovaLabs", city="Bengaluru",
         work_mode="hybrid", exp_min=0, exp_max=2, salary_min_lpa=8, salary_max_lpa=12,
         skills=["Python", "FastAPI", "PostgreSQL"], description="Build REST APIs in python and fastapi."),
    dict(external_id="A2", title="Python Developer", company="ZenPay", city="Pune", work_mode="remote",
         exp_min=2, exp_max=5, salary_min_lpa=14, salary_max_lpa=20, skills=["python", "django", "redis"],
         description="Own python django services for payments."),
    dict(external_id="A3", title="React Developer", company="KiteWorks", city="Bangalore", work_mode="onsite",
         exp_min=1, exp_max=3, salary_min_lpa=9, salary_max_lpa=14, skills=["react", "typescript", "css"],
         description="Build the react frontend of our web app."),
    dict(external_id="A4", title="Data Analyst", company="ArcCloud", city="Gurugram", work_mode="onsite",
         exp_min=0, exp_max=1, salary_min_lpa=5, salary_max_lpa=7, skills=["sql", "excel", "power bi"],
         description="Build dashboards in power bi and sql."),
    dict(external_id="A5", title="LLM Engineer", company="NovaLabs", city="Bengaluru", work_mode="remote",
         exp_min=1, exp_max=3, salary_min_lpa=18, salary_max_lpa=30, skills=["python", "llm", "rag", "embeddings"],
         description="Build python agents with rag, embeddings and tool use."),
]


@pytest.fixture(scope="session", autouse=True)
def _database():
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    init_schema()
    yield
    from aeromatch.db import close_pool
    close_pool()


@pytest.fixture
def conn():
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as c:
        register_vector(c)
        c.execute("TRUNCATE jobs, companies, users RESTART IDENTITY CASCADE")
        c.commit()
        yield c


@pytest.fixture
def seeded(conn):
    ingest(conn, [JobIn(**j) for j in JOBS])
    ids = {r["external_id"]: r["id"] for r in conn.execute("SELECT id, external_id FROM jobs").fetchall()}
    return conn, ids


@pytest.fixture
def user(conn):
    uid = conn.execute("INSERT INTO users (name) VALUES ('Asha') RETURNING id").fetchone()["id"]
    conn.commit()
    return uid
