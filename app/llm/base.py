"""LLM provider interface. Swappable; stub works offline, OpenAI-compatible when keyed."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroundedAnswer:
    title: str
    body_markdown: str
    key_points: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    abstained: bool = False
    grounded: bool = True


class LLMProvider:
    name: str = "base"

    def generate(self, question: str, passages: list[dict], doc_lookup: dict) -> GroundedAnswer:
        raise NotImplementedError
