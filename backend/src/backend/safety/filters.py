import re
from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Verdict(StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class ModerationResult:
    verdict: Verdict
    severity: Severity
    categories: tuple[str, ...] = field(default_factory=tuple)
    rationale: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.verdict == Verdict.BLOCK


_BLOCK_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:build|make|create)\b.*\b(?:bomb|explosive|weapon)\b", re.I), ("weapons",)),
    (re.compile(r"\b(?:instructions?|step[-\s]?by[-\s]?step)\b.*\b(?:bomb|explosive|weapon)\b", re.I), ("weapons",)),
    (re.compile(r"\b(?:kill|murder|suicide)\b", re.I), ("violence",)),
    (re.compile(r"\b(?:credit card|ssn|social security number)\b", re.I), ("privacy",)),
    (re.compile(r"\b(system|root|admin)?\s*password\b", re.I), ("privacy",)),
)

_WARN_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:hack|exploit)\b", re.I), ("questionable",)),
    (re.compile(r"\b(?:nsfw|explicit)\b", re.I), ("adult",)),
)


def check_patterns(text: str) -> ModerationResult | None:
    for pattern, categories in _BLOCK_PATTERNS:
        if pattern.search(text):
            return ModerationResult(Verdict.BLOCK, Severity.HIGH, categories, f"Pattern: {pattern.pattern}")

    for pattern, categories in _WARN_PATTERNS:
        if pattern.search(text):
            return ModerationResult(Verdict.WARN, Severity.MEDIUM, categories, f"Pattern: {pattern.pattern}")

    return None


def run_moderation(text: str, llm=None) -> ModerationResult:
    text = text.strip()
    if not text:
        return ModerationResult(Verdict.BLOCK, Severity.HIGH, rationale="Empty input")

    pattern_result = check_patterns(text)
    if pattern_result:
        return pattern_result

    if llm:
        llm_result = _run_llm_moderation(text, llm)
        if llm_result:
            return llm_result

    return ModerationResult(Verdict.ALLOW, Severity.LOW, rationale="No issues detected")


def _run_llm_moderation(text: str, llm) -> ModerationResult | None:
    prompt = f"""You are a content moderation classifier. Respond with:
verdict: <allow|warn|block>
severity: <low|medium|high>
categories: <comma separated>
rationale: <one sentence>

User message: \"\"\"{text}\"\"\""""

    try:
        response = llm.complete(prompt)
        return _parse_moderation_response(response.text)
    except Exception:
        return None


def _parse_moderation_response(text: str) -> ModerationResult | None:
    def extract(field: str, default: str = "") -> str:
        match = re.search(rf"{field}\s*:\s*(.+)", text, re.I)
        return match.group(1).strip() if match else default

    try:
        verdict = Verdict(extract("verdict", "allow"))
        severity = Severity(extract("severity", "low"))
        categories = tuple(c.strip() for c in extract("categories").split(",") if c.strip())
        rationale = extract("rationale") or None
        return ModerationResult(verdict, severity, categories, rationale)
    except ValueError:
        return None
