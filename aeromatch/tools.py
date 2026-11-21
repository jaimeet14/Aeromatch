"""Tools the agent can call. Arguments are Pydantic models, so the JSON schema sent to the LLM and
the validation of what comes back are generated from the same source."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from . import memory
from .search import SearchFilters, normalise_city, search


class SearchJobsArgs(SearchFilters):
    query: str = Field(..., min_length=2, description="What the user is looking for, in their words")
    limit: int = Field(5, ge=1, le=10)


class JobIdArgs(BaseModel):
    job_id: int


class UpdatePreferencesArgs(BaseModel):
    roles: list[str] | None = Field(None, description="Roles the user wants, e.g. ['backend engineer']")
    cities: list[str] | None = None
    work_modes: list[Literal["onsite", "hybrid", "remote"]] | None = None
    skills: list[str] | None = Field(None, description="Skills the user says they have")
    years_experience: int | None = Field(None, ge=0, le=40)
    min_salary_lpa: float | None = Field(None, ge=0)


class FinalAnswer(BaseModel):
    """The structured reply shown to the user. Must be the last tool call of every turn."""
    message: str = Field(..., min_length=1, description="Reply to the user, plain text")
    job_ids: list[int] = Field(default_factory=list,
                               description="IDs of jobs recommended in the message; only IDs returned by tools")
    follow_up_question: str | None = Field(None, description="One question to refine the search, if useful")


@dataclass
class ToolContext:
    conn: Any
    user_id: int
    seen_job_ids: set[int] = field(default_factory=set)


def _search_jobs(ctx: ToolContext, a: SearchJobsArgs) -> dict:
    prefs = memory.get_preferences(ctx.conn, ctx.user_id)
    # Fall back to remembered preferences for any filter the model left empty.
    filters = SearchFilters(
        city=a.city or (prefs["cities"][0] if prefs and len(prefs["cities"]) == 1 else None),
        work_mode=a.work_mode or (prefs["work_modes"][0] if prefs and len(prefs["work_modes"]) == 1 else None),
        years_experience=a.years_experience if a.years_experience is not None
        else (prefs["years_experience"] if prefs else None),
        min_salary_lpa=a.min_salary_lpa if a.min_salary_lpa is not None
        else (float(prefs["min_salary_lpa"]) if prefs and prefs["min_salary_lpa"] is not None else None),
    )
    results = search(ctx.conn, a.query, filters, limit=a.limit, user_id=ctx.user_id)
    ctx.seen_job_ids.update(r["id"] for r in results)
    return {"applied_filters": filters.model_dump(exclude_none=True),
            "results": [{k: r[k] for k in ("id", "title", "company", "city", "work_mode", "exp_min",
                                           "exp_max", "salary_min_lpa", "salary_max_lpa", "matched_skills")}
                        for r in results]}


def _get_job(ctx: ToolContext, a: JobIdArgs) -> dict:
    row = ctx.conn.execute(
        """SELECT j.id, j.title, c.name AS company, j.description, j.skills, j.city, j.work_mode,
                  j.exp_min, j.exp_max, j.salary_min_lpa::float, j.salary_max_lpa::float, j.is_active
           FROM jobs j JOIN companies c ON c.id = j.company_id WHERE j.id = %s""", [a.job_id]).fetchone()
    if row is None:
        return {"error": f"job {a.job_id} not found"}
    ctx.seen_job_ids.add(row["id"])
    return row


def _update_preferences(ctx: ToolContext, a: UpdatePreferencesArgs) -> dict:
    changes = a.model_dump(exclude_none=True)
    if "cities" in changes:
        changes["cities"] = [normalise_city(c) for c in changes["cities"]]
    return {"preferences": memory.update_preferences(ctx.conn, ctx.user_id, changes)}


def _record(action: str) -> Callable[[ToolContext, JobIdArgs], dict]:
    def run(ctx: ToolContext, a: JobIdArgs) -> dict:
        if not memory.record_action(ctx.conn, ctx.user_id, a.job_id, action):
            return {"error": f"job {a.job_id} not found"}
        return {"ok": True, "job_id": a.job_id, "action": action}
    return run


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[ToolContext, Any], dict] | None  # None = terminal (final_answer)

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.args.model_json_schema()}}


TOOLS = {t.name: t for t in [
    Tool("search_jobs", "Semantic + keyword search over live job postings. Use the user's words as query "
         "and fill filters only when the user stated them.", SearchJobsArgs, _search_jobs),
    Tool("get_job_details", "Full description, skills and salary for one job.", JobIdArgs, _get_job),
    Tool("update_preferences", "Remember durable facts the user states about themselves (skills, experience, "
         "preferred cities, work mode, roles, minimum salary).", UpdatePreferencesArgs, _update_preferences),
    Tool("save_job", "Save a job to the user's shortlist when they ask to.", JobIdArgs, _record("saved")),
    Tool("dismiss_job", "Hide a job the user is not interested in from future results.", JobIdArgs,
         _record("dismissed")),
    Tool("final_answer", "Reply to the user. Always end the turn with exactly one call to this tool.",
         FinalAnswer, None),
]}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


def execute(ctx: ToolContext, name: str, raw_args: str | dict) -> tuple[str, bool]:
    """Run a tool. Returns (json result for the model, ok). Errors go back to the model to self-correct."""
    tool = TOOLS.get(name)
    if tool is None or tool.run is None:
        return json.dumps({"error": f"unknown tool {name!r}"}), False
    try:
        args = tool.args.model_validate_json(raw_args) if isinstance(raw_args, str) \
            else tool.args.model_validate(raw_args)
    except ValidationError as e:
        return json.dumps({"error": "invalid arguments", "details": e.errors(include_url=False)},
                          default=str), False
    result = tool.run(ctx, args)
    return json.dumps(result, default=str), "error" not in result
