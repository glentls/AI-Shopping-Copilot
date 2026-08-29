from __future__ import annotations

import copy
import json
import threading
import warnings
from contextlib import contextmanager
from typing import Generator


ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

# Stable priority order for clarification questions — most discriminating first.
ATTRIBUTE_PRIORITY = (
    "category", "use_case", "material", "color", "size",
    "style", "brand", "budget", "feature", "other",
)


def _empty_entry(session_id: str, user_profile: dict) -> dict:
    return {
        "session_id": session_id,
        "user_profile": copy.deepcopy(user_profile),
        "turn": 0,
        "intent": None,          # "buying" | "browsing" | "intent_override" | "boundary"
        "constraints": {},       # {attribute: value} hard filters → Nick's BM25 / John's reranker
        "soft_preferences": [],  # free-text product-type keywords ("boots", "winter jacket")
        "asked_attributes": [],  # ordered list of attributes already asked → Shreya's clarifier
        "query_string": "",      # assembled by build_and_store_query() → Nick reads this
    }


def build_query(state: dict) -> str:
    """Assemble a BM25 query string from a session state snapshot.

    Soft preferences (product type) come first so BM25 weights the core
    product term most heavily. Hard constraint values follow.
    """
    parts: list[str] = list(state["soft_preferences"])
    parts.extend(state["constraints"].values())
    return " ".join(parts).strip()


class LedgerService:
    """Thread-safe, in-memory session ledger.

    Each session gets its own RLock so concurrent sessions never block each
    other, and the context manager holds the lock across the full
    read-yield-write cycle to eliminate TOCTOU races.

    Typical usage:
        ledger = LedgerService()
        ledger.create(session_id, user_profile)

        with ledger.session(session_id) as state:
            state["turn"] += 1
            state["constraints"]["color"] = "black"
        # written back atomically on clean exit
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._global_lock = threading.Lock()  # guards _store and _locks dicts only

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, session_id: str, user_profile: dict) -> None:
        """Initialise a fresh ledger entry. Overwrites any existing entry."""
        with self._global_lock:
            self._locks[session_id] = threading.RLock()
            self._store[session_id] = _empty_entry(session_id, user_profile)

    def read(self, session_id: str) -> dict:
        """Return a deep copy of the session state (safe to mutate freely)."""
        lock = self._session_lock(session_id)
        with lock:
            return copy.deepcopy(self._store[session_id])

    def delete(self, session_id: str) -> None:
        """Remove the session from the ledger."""
        with self._global_lock:
            self._store.pop(session_id, None)
            self._locks.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        with self._global_lock:
            return session_id in self._store

    # ------------------------------------------------------------------
    # Context manager — hold per-session lock for entire read-modify-write
    # ------------------------------------------------------------------

    @contextmanager
    def session(self, session_id: str) -> Generator[dict, None, None]:
        """Yield a mutable snapshot; commit it back atomically on clean exit.

        The per-session lock is held for the entire duration so no other
        call can interleave between the read and the write-back.

        Usage:
            with ledger.session(session_id) as state:
                state["turn"] += 1
                state["constraints"]["material"] = "leather"
        """
        lock = self._session_lock(session_id)
        with lock:
            snapshot = copy.deepcopy(self._store[session_id])
            try:
                yield snapshot
            except Exception:
                raise  # store is left unchanged on error
            else:
                self._store[session_id] = copy.deepcopy(snapshot)

    # ------------------------------------------------------------------
    # Fine-grained helpers — teammates should use these, not a generic update()
    # ------------------------------------------------------------------

    def increment_turn(self, session_id: str) -> None:
        with self._session_lock(session_id):
            self._store[session_id]["turn"] += 1

    def set_intent(self, session_id: str, intent: str) -> None:
        with self._session_lock(session_id):
            self._store[session_id]["intent"] = intent

    def add_constraint(self, session_id: str, attribute: str, value: str) -> None:
        """Add or overwrite a hard constraint. Warns if attribute is not in ALLOWED_ATTRIBUTES."""
        if attribute not in ALLOWED_ATTRIBUTES:
            warnings.warn(f"Unknown attribute '{attribute}', storing under 'other'.")
            attribute = "other"
        with self._session_lock(session_id):
            self._store[session_id]["constraints"][attribute] = value

    def clear_constraints(self, session_id: str) -> None:
        """Wipe all constraints and soft preferences — call this on intent override."""
        with self._session_lock(session_id):
            self._store[session_id]["constraints"].clear()
            self._store[session_id]["soft_preferences"].clear()

    def add_soft_preference(self, session_id: str, preference: str) -> None:
        with self._session_lock(session_id):
            prefs = self._store[session_id]["soft_preferences"]
            if preference not in prefs:
                prefs.append(preference)

    def mark_attribute_asked(self, session_id: str, attribute: str) -> None:
        with self._session_lock(session_id):
            asked = self._store[session_id]["asked_attributes"]
            if attribute not in asked:
                asked.append(attribute)

    def set_query_string(self, session_id: str, query: str) -> None:
        with self._session_lock(session_id):
            self._store[session_id]["query_string"] = query

    def build_and_store_query(self, session_id: str) -> str:
        """Build the BM25 query from current state and persist it. Returns the query string."""
        with self._session_lock(session_id):
            state = self._store[session_id]
            query = build_query(state)
            state["query_string"] = query
        return query

    def next_unasked_attribute(self, session_id: str) -> str | None:
        """Return the highest-priority attribute not yet asked or constrained, or None."""
        with self._session_lock(session_id):
            asked = set(self._store[session_id]["asked_attributes"])
            constraints = set(self._store[session_id]["constraints"].keys())
        covered = asked | constraints
        for attr in ATTRIBUTE_PRIORITY:
            if attr not in covered:
                return attr
        return None

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def dump(self, session_id: str) -> None:
        """Pretty-print a session's state to stdout."""
        print(json.dumps(self.read(session_id), indent=2))

    def __repr__(self) -> str:
        with self._global_lock:
            sessions = list(self._store.keys())
        return f"LedgerService(sessions={sessions})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _session_lock(self, session_id: str) -> threading.RLock:
        with self._global_lock:
            if session_id not in self._locks:
                raise KeyError(f"Session '{session_id}' not found in ledger.")
            return self._locks[session_id]
