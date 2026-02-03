import json
import re
from dataclasses import dataclass
from enum import StrEnum


class Intent(StrEnum):
    QA = "qa"
    SMALL_TALK = "small_talk"
    FINANCE_QUOTE = "finance_quote"
    SEARCH = "search"
    MEMORY_WRITE = "memory_write"
    ESCALATE = "escalate"
    BAD = "bad"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    rationale: str | None = None
    raw: str | None = None

    @property
    def needs_retrieval(self) -> bool:
        return self.intent in {Intent.QA, Intent.SEARCH}


_BAD_PATTERNS = [
    re.compile(r"\b(system|root|admin)?\s*password\b", re.I),
    re.compile(r"\bshare\s+(?:your|the)\s+(?:credentials|password|secret)\b", re.I),
]

_SMALL_TALK_PATTERNS = [
    re.compile(r"\b(hi|hello|hey|howdy)\b", re.I),
    re.compile(r"\b(how are you|what's up|whats up)\b", re.I),
    re.compile(r"\b(thank(s| you)|appreciate)\b", re.I),
]

_MEMORY_WRITE_PATTERNS = [
    re.compile(r"\bremember that\b", re.I),
    re.compile(r"\bsave (this|that|my)\b", re.I),
]

_FINANCE_PATTERNS = [
    re.compile(r"\b(stock|share)s?\s+(?:price|quote)\b", re.I),
    re.compile(r"\b(?:price|quote)\s+(?:for|of)\s+[A-Za-z]{1,5}\b", re.I),
    re.compile(r"\bticker\b", re.I),
]

_SEARCH_PATTERNS = [
    re.compile(r"\bgoogle\b", re.I),
    re.compile(r"\bsearch for\b", re.I),
    re.compile(r"\blook up\b", re.I),
]

_SYMBOL_PATTERNS = [
    re.compile(r"\b(?:price|quote)\s+(?:for|of)\s+([A-Za-z]{1,5})\b", re.I),
    re.compile(r"\b([A-Za-z]{1,5})\s+(?:stock|share)s?\s+(?:price|quote)\b", re.I),
    re.compile(r"\bticker\s+([A-Za-z]{1,5})\b", re.I),
]


def extract_symbol(text: str) -> str | None:
    for pattern in _SYMBOL_PATTERNS:
        if match := pattern.search(text):
            return match.group(1).upper()
    return None


def classify_heuristic(text: str) -> IntentResult | None:
    for p in _BAD_PATTERNS:
        if p.search(text):
            return IntentResult(Intent.BAD, "Credential harvesting")

    for p in _SMALL_TALK_PATTERNS:
        if p.search(text):
            return IntentResult(Intent.SMALL_TALK, "Greeting detected")

    for p in _MEMORY_WRITE_PATTERNS:
        if p.search(text):
            return IntentResult(Intent.MEMORY_WRITE, "Memory request")

    for p in _SEARCH_PATTERNS:
        if p.search(text):
            return IntentResult(Intent.SEARCH, "Search request")

    if extract_symbol(text) and any(p.search(text) for p in _FINANCE_PATTERNS):
        return IntentResult(Intent.FINANCE_QUOTE, f"Finance quote for {extract_symbol(text)}")

    if len(text.split()) <= 3 and text.endswith("?"):
        return IntentResult(Intent.SMALL_TALK, "Short question")

    return None


def classify_intent(text: str, llm=None) -> IntentResult:
    text = text.strip()
    if not text:
        return IntentResult(Intent.BAD, "Empty input")

    if result := classify_heuristic(text):
        return result

    if llm:
        if result := _classify_with_llm(text, llm):
            return result

    return IntentResult(Intent.QA, "Default fallback")


def _classify_with_llm(text: str, llm) -> IntentResult | None:
    prompt = f"""Classify intent as JSON: {{"intent": "<qa|small_talk|finance_quote|search|memory_write|escalate|bad>", "rationale": "<reason>"}}

User: \"\"\"{text}\"\"\"

JSON:"""

    try:
        response = llm.complete(prompt)
        return _parse_intent_response(response.text)
    except Exception:
        return None


def _parse_intent_response(text: str) -> IntentResult | None:
    if "```" in text:
        start, end = text.find("```"), text.rfind("```")
        if start != -1 and end > start:
            text = text[start + 3:end].replace("json", "", 1).strip()

    try:
        data = json.loads(text)
        intent = Intent(data.get("intent"))
        return IntentResult(intent, data.get("rationale"), text)
    except (json.JSONDecodeError, ValueError):
        return None
