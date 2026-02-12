import re

from .state import Moderation, Verdict


BLOCK_PATTERNS = [
    (r"\b(?:build|make|create)\b.*\b(?:bomb|explosive|weapon)\b", "weapons"),
    (r"\b(?:kill|murder|suicide)\b", "violence"),
    (r"\b(system|root|admin)?\s*password\b", "credentials"),
]

WARN_PATTERNS = [
    (r"\b(?:hack|exploit)\b", "security"),
]


class Moderator:
    def moderate(self, text: str) -> Moderation:
        if not text or not text.strip():
            return Moderation(Verdict.BLOCK, "Empty input")

        text_lower = text.strip().lower()

        for pattern, reason in BLOCK_PATTERNS:
            if re.search(pattern, text_lower):
                return Moderation(Verdict.BLOCK, reason)

        for pattern, reason in WARN_PATTERNS:
            if re.search(pattern, text_lower):
                return Moderation(Verdict.WARN, reason)

        return Moderation(Verdict.ALLOW)
