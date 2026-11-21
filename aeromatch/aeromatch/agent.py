"""The conversational job-matching agent: a bounded tool-use loop with persistent memory and a
validated, grounded structured reply."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import memory
from .config import settings
from .llm import LLM
from .tools import FinalAnswer, ToolContext, execute, tool_schemas

SYSTEM_PROMPT = """You are AeroMatch, a career agent that helps people in India find jobs through conversation.

Rules:
- When the user states a durable fact about themselves (skills, years of experience, city, work mode,
  target role, minimum salary), call update_preferences before anything else.
- Before recommending jobs, call search_jobs. Only fill a filter if the user stated it or it is in
  their saved preferences below.
- Never invent jobs. Recommend only job IDs returned by a tool in this conversation.
- End every turn with exactly one final_answer call: a short message, the recommended job_ids, and
  at most one follow-up question.

What you already know about this user:
{preferences}"""

NUDGE = "Respond by calling the final_answer tool."
_ID_RE = re.compile(r'"id":\s*(\d+)')


@dataclass
class TurnResult:
    conversation_id: int
    answer: FinalAnswer
    trace: list[dict] = field(default_factory=list)
    steps: int = 0
    repairs: int = 0          # invalid tool args / ungrounded or malformed final answers sent back
    structured: bool = True   # False if we had to fall back from free text
    latency_ms: float = 0.0


def _system(conn, user_id: int) -> dict:
    prefs = memory.get_preferences(conn, user_id)
    known = {k: v for k, v in (prefs or {}).items() if v not in (None, [])}
    return {"role": "system", "content": SYSTEM_PROMPT.format(
        preferences=json.dumps(known, default=float) if known else "nothing yet")}


def _check_final(raw: str, seen: set[int]) -> tuple[FinalAnswer | None, str | None]:
    try:
        ans = FinalAnswer.model_validate_json(raw)
    except ValidationError as e:
        return None, json.dumps({"error": "final_answer did not match the schema",
                                 "details": e.errors(include_url=False)}, default=str)
    invented = sorted(set(ans.job_ids) - seen)
    if invented:
        return None, json.dumps({"error": f"job_ids {invented} were not returned by any tool. "
                                          "Search first or remove them."})
    return ans, None


def run_turn(conn, llm: LLM, user_id: int, message: str, conversation_id: int | None = None) -> TurnResult:
    start = time.perf_counter()
    cid = memory.get_or_create_conversation(conn, user_id, conversation_id)
    history = memory.load_history(conn, cid, settings.history_messages)
    ctx = ToolContext(conn, user_id, {int(i) for m in history if m["role"] == "tool"
                                      for i in _ID_RE.findall(m["content"] or "")})

    user_msg = {"role": "user", "content": message}
    memory.append_message(conn, cid, user_msg)
    messages = [_system(conn, user_id), *history, user_msg]
    result = TurnResult(cid, FinalAnswer(message="Sorry, I couldn't finish that. Could you rephrase?"))
    nudged = False

    for _ in range(settings.max_agent_steps):
        result.steps += 1
        resp = llm.complete(messages, tool_schemas())

        if not resp.tool_calls:
            if not nudged and resp.content is not None:
                nudged = True
                messages += [resp.as_message(), {"role": "user", "content": NUDGE}]
                continue
            result.answer = FinalAnswer(message=resp.content or result.answer.message)
            result.structured = False
            memory.append_message(conn, cid, {"role": "assistant", "content": result.answer.message})
            break

        assistant = resp.as_message()
        messages.append(assistant)
        memory.append_message(conn, cid, assistant)

        final = None
        for call in resp.tool_calls:
            if call.name == "final_answer":
                final, error = _check_final(call.arguments, ctx.seen_job_ids)
                output, ok = (json.dumps({"ok": True}), True) if final else (error, False)
            else:
                output, ok = execute(ctx, call.name, call.arguments)
            result.repairs += not ok
            result.trace.append({"tool": call.name, "arguments": call.arguments, "ok": ok, "output": output[:600]})
            tool_msg = {"role": "tool", "tool_call_id": call.id, "content": output}
            messages.append(tool_msg)
            memory.append_message(conn, cid, tool_msg)

        if final is not None:
            result.answer = final
            break

    conn.commit()
    result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return result
