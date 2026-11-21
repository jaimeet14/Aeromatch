import json

from aeromatch import memory
from aeromatch.tools import FinalAnswer, ToolContext, execute, tool_schemas


def test_schemas_are_generated_from_pydantic():
    names = {s["function"]["name"] for s in tool_schemas()}
    assert names == {"search_jobs", "get_job_details", "update_preferences", "save_job", "dismiss_job",
                     "final_answer"}
    search = next(s for s in tool_schemas() if s["function"]["name"] == "search_jobs")["function"]
    assert search["parameters"]["required"] == ["query"]
    assert set(search["parameters"]["properties"]["work_mode"]["anyOf"][0]["enum"]) == {"onsite", "hybrid", "remote"}


def test_invalid_arguments_are_returned_not_raised(seeded, user):
    conn, _ = seeded
    out, ok = execute(ToolContext(conn, user), "search_jobs", '{"query": "x", "work_mode": "office"}')
    assert not ok and json.loads(out)["error"] == "invalid arguments"
    out, ok = execute(ToolContext(conn, user), "search_jobs", "not json")
    assert not ok
    out, ok = execute(ToolContext(conn, user), "delete_everything", "{}")
    assert not ok and "unknown tool" in out


def test_search_tracks_seen_ids_and_uses_remembered_preferences(seeded, user):
    conn, ids = seeded
    memory.update_preferences(conn, user, {"cities": ["Pune"], "years_experience": 3})
    ctx = ToolContext(conn, user)
    out, ok = execute(ctx, "search_jobs", {"query": "python developer"})
    res = json.loads(out)
    assert ok and res["applied_filters"] == {"city": "Pune", "years_experience": 3}
    assert [r["id"] for r in res["results"]] == [ids["A2"]]
    assert ctx.seen_job_ids == {ids["A2"]}


def test_update_preferences_merges_and_normalises(conn, user):
    ctx = ToolContext(conn, user)
    execute(ctx, "update_preferences", {"skills": ["Python", "SQL"], "cities": ["bangalore"]})
    out, _ = execute(ctx, "update_preferences", {"skills": ["python", "react"], "years_experience": 1})
    prefs = json.loads(out)["preferences"]
    assert prefs["skills"] == ["python", "sql", "react"]
    assert prefs["cities"] == ["Bengaluru"]
    assert prefs["years_experience"] == 1


def test_save_and_dismiss(seeded, user):
    conn, ids = seeded
    ctx = ToolContext(conn, user)
    assert execute(ctx, "save_job", {"job_id": ids["A1"]})[1]
    assert not execute(ctx, "save_job", {"job_id": 999999})[1]
    assert [j["id"] for j in memory.saved_jobs(conn, user)] == [ids["A1"]]
    execute(ctx, "dismiss_job", {"job_id": ids["A1"]})  # changing your mind overwrites the action
    assert memory.saved_jobs(conn, user) == []


def test_final_answer_schema():
    assert FinalAnswer.model_validate({"message": "hi"}).job_ids == []
