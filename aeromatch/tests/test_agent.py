import json

from aeromatch import memory
from aeromatch.agent import NUDGE, run_turn
from aeromatch.llm import LLMResponse, ScriptedLLM, ToolCall


def call(name, i=0, **args):
    return LLMResponse(None, [ToolCall(f"c{name}{i}", name, json.dumps(args))])


def result_ids(messages):
    """Job ids from the last search_jobs tool result in the transcript."""
    tool = [m for m in messages if m["role"] == "tool" and '"results"' in (m["content"] or "")][-1]
    return [r["id"] for r in json.loads(tool["content"])["results"]]


def test_search_then_grounded_answer(seeded, user):
    conn, ids = seeded

    def script(msgs):
        if msgs[-1]["role"] == "user":
            return call("search_jobs", query="python backend", city="Bengaluru")
        return call("final_answer", message="Here you go", job_ids=result_ids(msgs)[:2])

    r = run_turn(conn, ScriptedLLM(script), user, "python backend jobs in bangalore")
    assert r.structured and r.repairs == 0 and r.steps == 2
    assert r.answer.job_ids and set(r.answer.job_ids) <= {ids["A1"], ids["A5"]}
    assert [t["tool"] for t in r.trace] == ["search_jobs", "final_answer"]
    roles = [m["role"] for m in memory.load_history(conn, r.conversation_id, 50)]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]


def test_invented_job_ids_are_rejected_then_repaired(seeded, user):
    conn, _ = seeded
    llm = ScriptedLLM([call("final_answer", message="Try job 424242", job_ids=[424242]),
                       call("search_jobs", query="python"),
                       call("final_answer", 1, message="ok", job_ids=[])])
    r = run_turn(conn, llm, user, "anything for me?")
    assert r.repairs == 1 and r.trace[0]["ok"] is False
    assert "were not returned by any tool" in llm.calls[1][-1]["content"]
    assert r.answer.message == "ok"


def test_invalid_tool_arguments_go_back_to_the_model(seeded, user):
    conn, _ = seeded
    llm = ScriptedLLM([call("search_jobs", query="python", work_mode="wfh"),
                       call("search_jobs", 1, query="python", work_mode="remote"),
                       call("final_answer", message="done")])
    r = run_turn(conn, llm, user, "remote python roles")
    assert [t["ok"] for t in r.trace] == [False, True, True] and r.repairs == 1


def test_plain_text_is_nudged_into_structured_output(seeded, user):
    conn, _ = seeded
    llm = ScriptedLLM([LLMResponse("Sure, what city?"), call("final_answer", message="What city?")])
    r = run_turn(conn, llm, user, "hi")
    assert r.structured and llm.calls[1][-1]["content"] == NUDGE


def test_unstructured_fallback_when_model_ignores_the_nudge(seeded, user):
    conn, _ = seeded
    r = run_turn(conn, ScriptedLLM([LLMResponse("hello"), LLMResponse("hello again")]), user, "hi")
    assert not r.structured and r.answer.message == "hello again"


def test_step_budget_is_enforced(seeded, user):
    conn, _ = seeded
    r = run_turn(conn, ScriptedLLM(lambda m: call("search_jobs", query="python")), user, "loop forever")
    assert r.steps == 6 and "couldn't finish" in r.answer.message


def test_memory_persists_across_turns(seeded, user):
    conn, ids = seeded
    turn1 = ScriptedLLM([call("update_preferences", cities=["Pune"], skills=["Django"]),
                         call("search_jobs", query="python"),
                         call("final_answer", message="Found one", job_ids=[ids["A2"]])])
    r1 = run_turn(conn, turn1, user, "I'm in Pune and know Django")
    assert r1.answer.job_ids == [ids["A2"]]

    # New turn, same conversation: preferences are in the system prompt, history is replayed,
    # and IDs surfaced in turn 1 are still valid to recommend.
    turn2 = ScriptedLLM([call("save_job", job_id=ids["A2"]),
                         call("final_answer", message="Saved", job_ids=[ids["A2"]])])
    r2 = run_turn(conn, turn2, user, "save that one", r1.conversation_id)
    sent = turn2.calls[0]
    assert '"Pune"' in sent[0]["content"] and '"django"' in sent[0]["content"]
    assert sent[1] == {"role": "user", "content": "I'm in Pune and know Django"}
    assert r2.repairs == 0 and r2.conversation_id == r1.conversation_id
    assert [j["id"] for j in memory.saved_jobs(conn, user)] == [ids["A2"]]


def test_conversations_are_scoped_to_their_user(seeded, user):
    conn, _ = seeded
    other = memory.create_user(conn, "Ravi")
    r = run_turn(conn, ScriptedLLM([call("final_answer", message="hi")]), user, "hi")
    r2 = run_turn(conn, ScriptedLLM([call("final_answer", message="hi")]), other, "hi", r.conversation_id)
    assert r2.conversation_id != r.conversation_id
