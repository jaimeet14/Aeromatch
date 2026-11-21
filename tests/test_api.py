import json

import pytest
from fastapi.testclient import TestClient

from aeromatch.llm import LLMResponse, ScriptedLLM, ToolCall
from aeromatch.main import app, get_llm


@pytest.fixture
def client(seeded):
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_and_search(client):
    assert client.get("/health").json() == {"status": "ok", "active_jobs": 5}
    r = client.get("/search", params={"q": "react frontend", "city": "bangalore"})
    assert r.status_code == 200 and r.json()[0]["title"] == "React Developer"
    assert client.get("/search", params={"q": "x"}).status_code == 422


def test_chat_round_trip(client, seeded):
    _, ids = seeded
    script = [LLMResponse(None, [ToolCall("1", "search_jobs", json.dumps({"query": "data analyst sql"}))]),
              LLMResponse(None, [ToolCall("2", "final_answer", json.dumps(
                  {"message": "One match", "job_ids": [ids["A4"]], "follow_up_question": "Open to Pune?"}))])]
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM(script)
    uid = client.post("/users", json={"name": "Asha"}).json()["id"]
    r = client.post("/chat", json={"user_id": uid, "message": "data analyst roles"})
    body = r.json()
    assert r.status_code == 200
    assert body["reply"]["follow_up_question"] == "Open to Pune?"
    assert [j["title"] for j in body["jobs"]] == ["Data Analyst"]


def test_chat_unknown_user(client):
    app.dependency_overrides[get_llm] = lambda: ScriptedLLM([])
    assert client.post("/chat", json={"user_id": 999, "message": "hi"}).status_code == 404
