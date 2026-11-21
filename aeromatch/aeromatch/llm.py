"""Thin LLM client over any OpenAI-compatible chat endpoint, plus a scripted fake for tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .config import settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, validated by the tool layer


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls]
        return msg


class LLM(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAICompatibleLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url or settings.llm_base_url, api_key=api_key or settings.llm_api_key)
        self._model = model or settings.llm_model
        self._temperature = temperature

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        r = self._client.chat.completions.create(model=self._model, messages=messages, tools=tools,
                                                 tool_choice="auto", temperature=self._temperature)
        m = r.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (m.tool_calls or [])]
        return LLMResponse(m.content, calls)


class ScriptedLLM:
    """Replays a fixed list of responses (or calls a function) — deterministic agent tests."""

    def __init__(self, script: list[LLMResponse] | Callable[[list[dict]], LLMResponse]) -> None:
        self._script = script
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if callable(self._script):
            return self._script(messages)
        return self._script.pop(0)
