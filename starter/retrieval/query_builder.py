from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "still", "exploring", "matters", "requirement", "those", "options", "quite",
    "right", "yet", "ask", "about", "one", "specific", "attribute", "preference",
    "additional", "actually", "ignore", "earlier", "need", "what", "key",
}

BOILERPLATE_SUBSTRINGS = (
    "not quite right yet",
    "ask me about one specific attribute",
    "don't have a preference for",
    "don't have an additional preference for",
    "please use your judgment",
    "i'm looking for",
    "but i'm still exploring",
)

# Constraints legitimately contain periods ("3.5 inches", "100% cotton. Imported."),
# so each turn occupies its own line and captures run to the end of that line.
CONSTRAINT_PATTERNS = (
    re.compile(r"key requirement is:\s*(.+)", re.I),
    re.compile(r"what matters is:\s*(.+)", re.I),
    re.compile(r"what i need is:\s*(.+)", re.I),
    re.compile(r"budget around\s*(\$?\s*[\d.]+)", re.I),
)


def strip_boilerplate(query_text: str) -> str:
    cleaned = re.sub(r"[^\S\n]+", " ", query_text.strip())
    for phrase in BOILERPLATE_SUBSTRINGS:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.I)
    return re.sub(r"[^\S\n]+", " ", cleaned).strip()


def quote_fts(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" -;,.\t\n")
    if not cleaned:
        return ""
    escaped = cleaned.replace('"', '""')
    return f'"{escaped}"'


def extract_constraint_phrases(query_text: str) -> list[str]:
    phrases: list[str] = []
    for pattern in CONSTRAINT_PATTERNS:
        for match in pattern.finditer(query_text):
            captured = match.group(1).strip(" -;,.\t\n")
            if not captured:
                continue
            if ";" in captured:
                phrases.extend(part.strip(" -;,.\t\n") for part in captured.split(";") if part.strip())
            else:
                phrases.append(captured)
    return list(dict.fromkeys(phrase for phrase in phrases if phrase))


def tokenize_terms(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            token.lower()
            for token in TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in STOPWORDS
        )
    )[:40]


def _phrases_and_terms(
    query_text: str,
    slot_values: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    cleaned = strip_boilerplate(query_text)
    phrases = extract_constraint_phrases(cleaned)
    if slot_values:
        for value in slot_values:
            normalized = value.strip()
            if normalized and normalized not in phrases:
                phrases.append(normalized)
    phrases = list(dict.fromkeys(phrase for phrase in phrases if phrase))[:6]

    terms = tokenize_terms(cleaned)
    terms = [
        term
        for term in terms
        if not any(term in phrase.lower() for phrase in phrases)
    ]
    return phrases, terms


def build_fts_expression(
    query_text: str,
    mode: str = "browsing",
    slot_values: list[str] | None = None,
) -> str:
    phrases, terms = _phrases_and_terms(query_text, slot_values)
    phrase_clauses = [clause for phrase in phrases if (clause := quote_fts(phrase))]
    term_clauses = [clause for term in terms if (clause := quote_fts(term))]

    if mode == "buying":
        parts: list[str] = []
        if phrase_clauses:
            if len(phrase_clauses) == 1:
                parts.append(phrase_clauses[0])
            else:
                parts.append(" AND ".join(phrase_clauses))
        if term_clauses:
            term_part = " OR ".join(term_clauses[:8])
            if len(term_clauses) > 1:
                term_part = f"({term_part})"
            parts.append(term_part)
        if not parts:
            return ""
        return " AND ".join(parts) if len(parts) > 1 else parts[0]

    all_clauses = phrase_clauses + term_clauses
    if not all_clauses:
        return ""
    return " OR ".join(all_clauses)
