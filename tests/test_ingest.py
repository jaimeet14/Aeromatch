from aeromatch.ingest import JobIn, ingest
from tests.conftest import JOBS


def feed(**overrides):
    return [JobIn(**{**j, **overrides.get(j["external_id"], {})}) for j in JOBS]


def test_first_load_inserts_and_normalises(conn):
    s = ingest(conn, feed())
    assert (s.inserted, s.embedded) == (5, 5)
    row = conn.execute("SELECT city, skills, embedding FROM jobs WHERE external_id = 'A3'").fetchone()
    assert row["city"] == "Bengaluru"
    assert row["skills"] == ["css", "react", "typescript"]
    assert len(row["embedding"].to_list()) == 256


def test_rerun_is_a_noop(conn):
    ingest(conn, feed())
    s = ingest(conn, feed())
    assert (s.unchanged, s.embedded, s.inserted, s.updated) == (5, 0, 0, 0)


def test_only_changed_rows_are_reembedded(conn):
    ingest(conn, feed())
    s = ingest(conn, feed(A2={"salary_max_lpa": 25}))
    assert (s.updated, s.embedded, s.unchanged) == (1, 1, 4)
    assert conn.execute("SELECT salary_max_lpa::float AS v FROM jobs WHERE external_id='A2'").fetchone()["v"] == 25


def test_missing_jobs_deactivated_then_revived(conn):
    ingest(conn, feed())
    s = ingest(conn, feed()[:3])
    assert s.deactivated == 2
    s = ingest(conn, feed())
    assert (s.reactivated, s.embedded) == (2, 0)
    assert conn.execute("SELECT count(*) AS n FROM jobs WHERE is_active").fetchone()["n"] == 5


def test_partial_feed_does_not_deactivate(conn):
    ingest(conn, feed())
    s = ingest(conn, feed()[:1], full_snapshot=False)
    assert s.deactivated == 0
