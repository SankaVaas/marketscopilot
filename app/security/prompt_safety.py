"""
Lightweight, dependency-free prompt-safety and DLP checks.

These are pattern-based guards suitable for a portfolio/demo build. In a real
deployment you would layer in a proper moderation model or a service like
NeMo Guardrails / Llama Guard (also open-source and self-hostable, so it stays
on-theme with the zero-cost stack) rather than relying on regexes alone.
"""
import re

# Very small set of illustrative injection patterns -- expand in production.
_INJECTION_PATTERNS = [
    r"ignore (all|previous|the) (instructions|prompt)",
    r"disregard (all|previous) (instructions|rules)",
    r"you are now",
    r"reveal (your|the) system prompt",
]

_PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b(?:\+?\d{1,3})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "account_number": r"\b\d{8,17}\b",
}


def detect_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _INJECTION_PATTERNS)


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Returns (redacted_text, list_of_pii_types_found)."""
    found = []
    redacted = text
    for label, pattern in _PII_PATTERNS.items():
        if re.search(pattern, redacted):
            found.append(label)
            redacted = re.sub(pattern, f"[REDACTED_{label.upper()}]", redacted)
    return redacted, found
