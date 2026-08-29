from __future__ import annotations

import json #read product data
import re #patterns for finding words in a sentence
import sqlite3 #a tiny database that lives in memory
from pathlib import Path

#finds chunks of letters and numbers (basically words) and removes the unn
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
    "alloy", "silver", "gold", "sterling", "stainless", "platinum", "brass", "copper", "titanium",
)
COLOR_WORDS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")
STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
CATEGORY_WORDS = (
    "shoes", "sneakers", "boots", "sandals", "heels", "flats",
    "jacket", "coat", "sweater", "hoodie", "shirt", "blouse", "dress", "skirt", "pants", "jeans", "shorts",
    "necklace", "bracelet", "earrings", "ring", "watch", "jewelry",
    "bag", "belt", "hat", "scarf", "gloves", "socks",
)


MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.IGNORECASE)
COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_WORDS) + r")\b", re.IGNORECASE)
BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*(\d+)", re.IGNORECASE)
SIZE_NUM_RE = re.compile(r"\bsize\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
CATEGORY_RE = re.compile(r"\b(" + "|".join(CATEGORY_WORDS) + r")\b", re.IGNORECASE)
OVERRIDE_PHRASES = ("actually", "never mind", "nevermind", "instead", "forget that", "forget it", "ignore my earlier", "scratch that", "forget everything")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]: #"I want black shoes" -> ["black", "shoes"]
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]

#given a piece of text, decides which category of attrubute that text is talking abt
def _classify(text: str) -> str:
    """Mirrors the evaluator's own classify_constraint logic."""
    lowered = text.lower()
    if "budget" in lowered or BUDGET_RE.search(lowered):
        return "budget"
    if any(m in lowered for m in MATERIALS):
        return "material"
    if any(w in lowered for w in COLOR_WORDS):
        return "color"
    if any(w in lowered for w in SIZE_WORDS):
        return "size"
    if any(w in lowered for w in STYLE_WORDS):
        return "style"
    if any(w in lowered for w in USE_CASE_WORDS):
        return "use_case"
    return "feature"

#given a message, actually pulls out the specific values mentioned and returns them as dict
def _extract_slots(text: str) -> dict:
    lower = text.lower()
    found: dict[str, str] = {}
    category_match = CATEGORY_RE.search(lower)
    if category_match:
        found["category"] = category_match.group(1)
    material_match = MATERIAL_RE.search(lower)
    if material_match:
        found["material"] = material_match.group(1)
    color_match = COLOR_RE.search(lower)
    if color_match:
        found["color"] = color_match.group(1)
    size_match = SIZE_NUM_RE.search(lower)
    if size_match:
        found["size"] = size_match.group(1)
    budget_match = BUDGET_RE.search(lower)
    if budget_match:
        found["budget"] = budget_match.group(1)
    for word in USE_CASE_WORDS:
        if word in lower:
            found["use_case"] = word
            break
    for word in ("style", "fit", "sleeve", "neck"):
        if word in lower and word != "style":
            found["style"] = word
            break
    return found

#handles the case whereby the user changes their mind, erase all prev slots
def _is_override(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in OVERRIDE_PHRASES)



class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path) #get the data on this agent so that other methods can use later
        self.connection = sqlite3.connect(":memory:") #a databse js for the session
        self._sessions: set[str] = set() #each session is "a user having one convo with the agent"
        self._slots: dict [str,dict] ={} #per-session slot storage
        self._clarify_count: dict[str,int] = {} #tracks clarifying qns asked
        self._build_index() #immediately loads every product into the database
        self._asked_attributes: dict[str, set] = {}

    def _build_index(self) -> None: #loading the catlog into a search table
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None: #start new session
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)
        self._slots[session_id] = {
            attr: None for attr in ALLOWED_ATTRIBUTES if attr != "other"
        }
        self._clarify_count[session_id] = 0
        self._asked_attributes[session_id] = set()

    #problem to solve: the agent is only looking at the current user_message and doesnt look at what was said in the turns before
    def respond( #ans one turn of the conversation
        self,
        session_id: str,
        user_message: str, #what the shopper says
        turn: int, #1 through 10, imp: unused in this starter (a smarter agent might ask questions early and recommend later)
        #Early turns (like turn 1-2): the agent probably doesn't have enough information yet — maybe it should ask a clarifying question instead of guessing blindly ("What's your budget?" "What color are you looking for?")
        #Later turns (like turn 5+): by now, hopefully enough information has been gathered through the conversation, so the agent should stop asking and start recommending actual products — because turns are limited (remember, max 10, and running out = zero score), so at some point it needs to commit to an answer rather than keep asking more questions forever
        top_k: int, #how many products to return
    ) -> dict:

        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        
        current_slots = self._slots[session_id]

        if _is_override(user_message):
            for key in current_slots:
                current_slots[key] = None
        
        new_info = _extract_slots(user_message)
        for key, value in new_info.items():
            current_slots[key] = value
        #print(f"DEBUG slots={current_slots}")
        
        combined_terms = list(dict.fromkeys(_terms(user_message)))
        for value in current_slots.values():
            if value:
                combined_terms.extend(_terms(str(value)))
        unique_terms = list(dict.fromkeys(combined_terms))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)

        if not expression:
            recommendations, candidate_count = [], 0
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?", #bm25 ranking
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
            candidate_count = self.connection.execute(
                "SELECT COUNT(*) FROM products WHERE products MATCH ?", (expression,)
            ).fetchone()[0]

        empty_slots = [k for k, v in current_slots.items() if v is None]
        clarifications_used = self._clarify_count[session_id]

        already_asked = self._asked_attributes[session_id]
        askable_slots = [s for s in empty_slots if s not in already_asked]

        should_clarify = bool(
            clarifications_used < 5
            and (len(empty_slots) >= 6 or candidate_count > 500)
            and askable_slots
        )
        #print(f"DEBUG turn={turn} msg={user_message!r} empty={len(empty_slots)} candidates={candidate_count} clarify={should_clarify}")

        if should_clarify:
            self._clarify_count[session_id] += 1
            ask = askable_slots[0]  # FIXED: use askable_slots, not empty_slots
            already_asked.add(ask)  # NEW: record that we asked about this
            return {
                "message": f"Could you tell me your preference for {ask}?",
                "ask_attribute": ask,
                "recommendations": recommendations,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    
