import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/aeromatch")
    # "wordllama" (offline, 256-d), "ollama:<model>" or "hash" (tests only)
    embedder: str = os.getenv("EMBEDDER", "wordllama")
    embed_dim: int = int(os.getenv("EMBED_DIM", "256"))
    # Any OpenAI-compatible endpoint: Ollama, OpenAI, Groq, vLLM ...
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "6"))
    history_messages: int = int(os.getenv("HISTORY_MESSAGES", "20"))


settings = Settings()
