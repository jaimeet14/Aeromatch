-- AeroMatch schema. {dim} is replaced with the embedding dimension at startup.
CREATE EXTENSION IF NOT EXISTS vector;

-- array_to_string is only STABLE; generated columns need IMMUTABLE.
CREATE OR REPLACE FUNCTION skills_text(text[]) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT array_to_string($1, ' ') $$;

CREATE TABLE IF NOT EXISTS companies (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS jobs (
    id              BIGSERIAL PRIMARY KEY,
    external_id     TEXT NOT NULL UNIQUE,              -- id from the source feed
    company_id      BIGINT NOT NULL REFERENCES companies(id),
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    skills          TEXT[] NOT NULL DEFAULT '{}',      -- lower-cased
    city            TEXT NOT NULL,
    work_mode       TEXT NOT NULL CHECK (work_mode IN ('onsite', 'hybrid', 'remote')),
    exp_min         SMALLINT NOT NULL CHECK (exp_min >= 0),
    exp_max         SMALLINT NOT NULL CHECK (exp_max >= exp_min),
    salary_min_lpa  NUMERIC(6,2),
    salary_max_lpa  NUMERIC(6,2),
    posted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    content_hash    TEXT NOT NULL,                     -- re-embed only when this changes
    embedding       vector({dim}),
    tsv             tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', title), 'A') ||
                        setweight(to_tsvector('english', skills_text(skills)), 'B') ||
                        setweight(to_tsvector('english', description), 'C')
                    ) STORED
);

CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw ON jobs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS jobs_tsv_gin        ON jobs USING gin (tsv);
CREATE INDEX IF NOT EXISTS jobs_skills_gin     ON jobs USING gin (skills);
CREATE INDEX IF NOT EXISTS jobs_active_filters ON jobs (city, work_mode, exp_min) WHERE is_active;

CREATE TABLE IF NOT EXISTS users (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Long-term memory: what the agent has learned about the user.
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id           BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    roles             TEXT[] NOT NULL DEFAULT '{}',
    cities            TEXT[] NOT NULL DEFAULT '{}',
    work_modes        TEXT[] NOT NULL DEFAULT '{}',
    skills            TEXT[] NOT NULL DEFAULT '{}',
    years_experience  SMALLINT,
    min_salary_lpa    NUMERIC(6,2),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Short-term memory: the conversation transcript, including tool calls.
CREATE TABLE IF NOT EXISTS conversations (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content          TEXT,
    tool_calls       JSONB,          -- assistant turns that requested tools
    tool_call_id     TEXT,           -- tool turns answering a call
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);

-- Explicit feedback, used for personalisation.
CREATE TABLE IF NOT EXISTS user_job_actions (
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id      BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    action      TEXT NOT NULL CHECK (action IN ('saved', 'dismissed')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, job_id)
);
