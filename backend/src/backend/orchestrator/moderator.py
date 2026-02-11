import re
from dataclasses import dataclass
from enum import StrEnum

from langgraph.types import Command


class Verdict(StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class Moderation:
    verdict: Verdict
    reason: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.verdict == Verdict.BLOCK


class Moderator:
    BLOCK_PATTERNS = [
        (r"\b(?:build|make|create)\b.*\b(?:bomb|explosive|weapon)\b", "weapons"),
        (r"\b(?:kill|murder|suicide)\b", "violence"),
        (r"\b(system|root|admin)?\s*password\b", "credentials"),
    ]

    WARN_PATTERNS = [
        (r"\b(?:hack|exploit)\b", "security"),
    ]

    async def __call__(self, state: dict) -> Command:
        moderation = self.moderate(state["query"])
        goto = "blocked" if moderation.is_blocked else "router"
        return Command(
            update={
                "is_blocked": moderation.is_blocked,
                "moderation_reason": moderation.reason,
            },
            goto=goto,
        )

    def moderate(self, text: str) -> Moderation:
        if not text or not text.strip():
            return Moderation(Verdict.BLOCK, "Empty input")

        text_lower = text.strip().lower()

        for pattern, reason in self.BLOCK_PATTERNS:
            if re.search(pattern, text_lower, re.I):
                return Moderation(Verdict.BLOCK, reason)

        for pattern, reason in self.WARN_PATTERNS:
            if re.search(pattern, text_lower, re.I):
                return Moderation(Verdict.WARN, reason)

        return Moderation(Verdict.ALLOW)
