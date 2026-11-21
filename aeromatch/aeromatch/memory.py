"""Persistence for the agent's two kinds of memory: the conversation transcript (short-term)
and the user's preferences + feedback (long-term)."""
from __future__ import annotations

import json

PREF_FIELDS = ("roles", "cities", "work_modes", "skills", "years_experience", "min_salary_lpa")
LIST_FIELDS = {"roles", "cities", "work_modes", "skills"}


def create_user(conn, name: str) -> int:
    uid = conn.execute("INSERT INTO users (name) VALUES (%s) RETURNING id", [name]).fetchone()["id"]
    conn.commit()
    return uid


def user_exists(conn, user_id: int) -> bool:
    return conn.execute("SELECT 1 FROM users WHERE id = %s", [user_id]).fetchone() is not None


def get_or_create_conversation(conn, user_id: int, conversation_id: int | None) -> int:
    if conversation_id is not None:
        row = conn.execute("SELECT id FROM conversations WHERE id = %s AND user_id = %s",
                           [conversation_id, user_id]).fetchone()
        if row:
            return row["id"]
    cid = conn.execute("INSERT INTO conversations (user_id) VALUES (%s) RETURNING id", [user_id]).fetchone()["id"]
    conn.commit()
    return cid


def append_message(conn, conversation_id: int, msg: dict) -> None:
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, tool_calls, tool_call_id) VALUES (%s,%s,%s,%s,%s)",
        [conversation_id, msg["role"], msg.get("content"),
         json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None, msg.get("tool_call_id")])


def load_history(conn, conversation_id: int, limit: int) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content, tool_calls, tool_call_id FROM messages WHERE conversation_id = %s "
        "ORDER BY id DESC LIMIT %s", [conversation_id, limit]).fetchall()[::-1]
    # Never start the window on an orphaned tool result.
    while rows and rows[0]["role"] == "tool":
        rows.pop(0)
    history = []
    for r in rows:
        m = {"role": r["role"], "content": r["content"]}
        if r["tool_calls"]:
            m["tool_calls"] = r["tool_calls"]
        if r["tool_call_id"]:
            m["tool_call_id"] = r["tool_call_id"]
        history.append(m)
    return history


def get_preferences(conn, user_id: int) -> dict | None:
    return conn.execute(f"SELECT {', '.join(PREF_FIELDS)} FROM user_preferences WHERE user_id = %s",
                        [user_id]).fetchone()


def update_preferences(conn, user_id: int, changes: dict) -> dict:
    """Merge: list fields are unioned, scalar fields overwritten."""
    current = get_preferences(conn, user_id) or {f: ([] if f in LIST_FIELDS else None) for f in PREF_FIELDS}
    merged = dict(current)
    for key, value in changes.items():
        if key in LIST_FIELDS:
            norm = [v.strip().lower() if key in ("skills", "roles") else v for v in value]
            merged[key] = list(dict.fromkeys([*current[key], *norm]))
        else:
            merged[key] = value
    conn.execute(
        f"""INSERT INTO user_preferences (user_id, {', '.join(PREF_FIELDS)}) VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (user_id) DO UPDATE SET
            {', '.join(f'{f} = EXCLUDED.{f}' for f in PREF_FIELDS)}, updated_at = now()""",
        [user_id, *[merged[f] for f in PREF_FIELDS]])
    if merged["min_salary_lpa"] is not None:
        merged["min_salary_lpa"] = float(merged["min_salary_lpa"])
    return merged


def record_action(conn, user_id: int, job_id: int, action: str) -> bool:
    if conn.execute("SELECT 1 FROM jobs WHERE id = %s", [job_id]).fetchone() is None:
        return False
    conn.execute(
        """INSERT INTO user_job_actions (user_id, job_id, action) VALUES (%s,%s,%s)
           ON CONFLICT (user_id, job_id) DO UPDATE SET action = EXCLUDED.action, created_at = now()""",
        [user_id, job_id, action])
    return True


def saved_jobs(conn, user_id: int) -> list[dict]:
    return conn.execute(
        """SELECT j.id, j.title, c.name AS company, j.city, j.work_mode
           FROM user_job_actions a JOIN jobs j ON j.id = a.job_id JOIN companies c ON c.id = j.company_id
           WHERE a.user_id = %s AND a.action = 'saved' ORDER BY a.created_at DESC""", [user_id]).fetchall()
