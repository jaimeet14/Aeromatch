from aeromatch import memory
from aeromatch.search import SearchFilters, normalise_city, rrf, search, vector_ids


def test_rrf_rewards_agreement_and_respects_weights():
    s = rrf([1, 2, 3], [3, 1, 4])
    assert max(s, key=s.get) == 1                       # high in both lists
    w = rrf([1, 2], [2, 1], weights=(0.1, 1.0))
    assert w[2] > w[1]                                  # the heavier ranking wins


def test_city_aliases():
    assert normalise_city("bangalore") == "Bengaluru"
    assert normalise_city(" Gurgaon ") == "Gurugram"
    assert normalise_city(None) is None
    assert SearchFilters(city="bombay").city == "Mumbai"


def test_filters_are_hard_constraints(seeded):
    conn, _ = seeded
    for f in [SearchFilters(city="bangalore"), SearchFilters(work_mode="remote"),
              SearchFilters(years_experience=0), SearchFilters(min_salary_lpa=15)]:
        for mode in ("hybrid", "vector", "lexical"):
            for r in search(conn, "python engineer", f, mode=mode):
                assert not f.city or r["city"] == f.city
                assert not f.work_mode or r["work_mode"] == f.work_mode
                assert f.years_experience is None or r["exp_min"] <= f.years_experience
                assert f.min_salary_lpa is None or r["salary_max_lpa"] >= f.min_salary_lpa


def test_prefiltered_vector_search_returns_every_match(seeded):
    conn, _ = seeded
    with conn.transaction():
        got = vector_ids(conn, "anything", SearchFilters(city="Bengaluru"))
    assert len(got) == 3  # A1, A3 (alias-normalised at ingest), A5


def test_lexical_matches_skill_terms(seeded):
    conn, ids = seeded
    top = search(conn, "power bi dashboards", mode="lexical")[0]
    assert top["id"] == ids["A4"]


def test_inactive_jobs_never_returned(seeded):
    conn, ids = seeded
    conn.execute("UPDATE jobs SET is_active = FALSE WHERE id = %s", [ids["A5"]])
    assert ids["A5"] not in {r["id"] for r in search(conn, "llm rag embeddings python", limit=10)}


def test_personalisation_excludes_dismissed_and_boosts_skills(seeded, user):
    conn, ids = seeded
    memory.record_action(conn, user, ids["A2"], "dismissed")
    memory.update_preferences(conn, user, {"skills": ["llm", "rag", "embeddings"]})
    results = search(conn, "python developer", limit=10, user_id=user)
    assert ids["A2"] not in {r["id"] for r in results}
    assert results[0]["id"] == ids["A5"]
    assert results[0]["matched_skills"] == ["embeddings", "llm", "rag"]
