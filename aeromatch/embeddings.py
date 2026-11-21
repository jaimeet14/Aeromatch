"""Pluggable text embedders. All return L2-normalised float32 vectors."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from .config import settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class WordLlamaEmbedder:
    """Static token embeddings distilled from an LLM; CPU-only, ~10k texts/s."""

    def __init__(self) -> None:
        from wordllama import WordLlama

        self._model = WordLlama.load(dim=256)
        self.dim = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._model.embed(texts))


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        import httpx

        self._client = httpx.Client(base_url=base_url, timeout=60)
        self._model = model
        self.dim = len(self.embed(["probe"])[0])

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self._client.post("/api/embed", json={"model": self._model, "input": texts})
        r.raise_for_status()
        return _normalise(r.json()["embeddings"])


class HashingEmbedder:
    """Deterministic bag-of-words hashing. For tests only — no semantics."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9+#]+", t.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        return _normalise(out + 1e-6)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    spec = settings.embedder
    if spec == "wordllama":
        return WordLlamaEmbedder()
    if spec == "hash":
        return HashingEmbedder(settings.embed_dim)
    if spec.startswith("ollama:"):
        return OllamaEmbedder(spec.split(":", 1)[1])
    raise ValueError(f"Unknown EMBEDDER {spec!r}")


def job_text(title: str, skills: list[str], description: str) -> str:
    """The text that represents a job in vector space."""
    return f"{title}. Skills: {', '.join(skills)}. {description}"
