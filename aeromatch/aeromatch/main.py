from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import memory
from .agent import run_turn
from .db import close_pool, connection, init_schema
from .llm import LLM, OpenAICompatibleLLM
from .search import SearchFilters, search
from .tools import FinalAnswer


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_schema()
    yield
    close_pool()


app = FastAPI(title="AeroMatch", version="1.0.0", lifespan=lifespan)
_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = OpenAICompatibleLLM()
    return _llm


class UserIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class ChatIn(BaseModel):
    user_id: int
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: int | None = None


class ChatOut(BaseModel):
    conversation_id: int
    reply: FinalAnswer
    jobs: list[dict]
    trace: list[dict]
    latency_ms: float


@app.get("/health")
def health():
    with connection() as conn:
        n = conn.execute("SELECT count(*) AS n FROM jobs WHERE is_active").fetchone()["n"]
    return {"status": "ok", "active_jobs": n}


@app.post("/users", status_code=201)
def create_user(body: UserIn):
    with connection() as conn:
        return {"id": memory.create_user(conn, body.name)}


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, llm: LLM = Depends(get_llm)):
    with connection() as conn:
        if not memory.user_exists(conn, body.user_id):
            raise HTTPException(404, "user not found")
        r = run_turn(conn, llm, body.user_id, body.message, body.conversation_id)
        jobs = conn.execute(
            """SELECT j.id, j.title, c.name AS company, j.city, j.work_mode
               FROM jobs j JOIN companies c ON c.id = j.company_id WHERE j.id = ANY(%s)""",
            [r.answer.job_ids]).fetchall()
    order = {jid: i for i, jid in enumerate(r.answer.job_ids)}
    return ChatOut(conversation_id=r.conversation_id, reply=r.answer, trace=r.trace, latency_ms=r.latency_ms,
                   jobs=sorted(jobs, key=lambda j: order[j["id"]]))


@app.get("/search")
def search_endpoint(q: str = Query(..., min_length=2), city: str | None = None,
                    work_mode: Literal["onsite", "hybrid", "remote"] | None = None,
                    years_experience: int | None = Query(None, ge=0, le=40),
                    min_salary_lpa: float | None = Query(None, ge=0),
                    user_id: int | None = None, limit: int = Query(10, ge=1, le=50),
                    mode: Literal["hybrid", "vector", "lexical"] = "hybrid"):
    f = SearchFilters(city=city, work_mode=work_mode, years_experience=years_experience,
                      min_salary_lpa=min_salary_lpa)
    with connection() as conn:
        return search(conn, q, f, limit=limit, user_id=user_id, mode=mode)


@app.get("/users/{user_id}/preferences")
def preferences(user_id: int):
    with connection() as conn:
        return memory.get_preferences(conn, user_id) or {}


@app.get("/users/{user_id}/saved")
def saved(user_id: int):
    with connection() as conn:
        return memory.saved_jobs(conn, user_id)
