from __future__ import annotations

from src.attributes import ascii_tokens
from src.catalog import Catalog
from src.contracts.retrieval import Candidate
from src.contracts.state import SessionState


# Selected once on the 120-session dev split. The public holdout is not used to
# revise this value.
PROFILE_RERANK_WEIGHT = 0.05
ALLOWED_PROFILE_WEIGHTS = frozenset((0.02, 0.05, 0.10))


class ProfileAffinityReranker:
    """Apply bounded within-session profile-tag affinity to supplied candidates.

    The public-set audit found the supplied profile to be near-signal-free:
    across all 200 public sessions only ``preference_tags`` and
    ``rating_style`` vary, while ``difficulty_bucket`` leaks scenario metadata.
    Profile use is therefore confined to the current session and retained only
    when supported by measured evaluation.

    Three properties keep that boundary structural rather than documentary:

    * The prior cannot add, remove, or resurrect a product. Under the default
      or evidence-scoped window it receives a frozen Top-K; a broader caller
      may expose a larger set whose later truncation can change membership.
    * Nothing is retained between sessions. The tags come from the profile the
      evaluator supplies to ``reset`` for this session alone.
    * The bonus is bounded, but remains additive and may reverse an incoming
      score gap smaller than that bound. It is a soft prior, not a hard
      relevance-ordering guarantee.
    """

    def __init__(
        self,
        catalog: Catalog,
        weight: float = PROFILE_RERANK_WEIGHT,
    ) -> None:
        if weight not in ALLOWED_PROFILE_WEIGHTS:
            raise ValueError(
                f"profile rerank weight must be one of {sorted(ALLOWED_PROFILE_WEIGHTS)}"
            )
        self.catalog = catalog
        self.weight = weight
        self._term_cache: dict[str, frozenset[str]] = {}

    def _product_terms(self, asin: str) -> frozenset[str]:
        cached = self._term_cache.get(asin)
        if cached is not None:
            return cached
        product = self.catalog.get(asin)
        value = (
            frozenset(ascii_tokens(product.searchable_text))
            if product is not None
            else frozenset()
        )
        self._term_cache[asin] = value
        return value

    @staticmethod
    def _profile_terms(state: SessionState) -> frozenset[str]:
        profile = state.user_profile
        if profile is None:
            return frozenset()
        terms: set[str] = set()
        for tag in profile.preference_tags:
            terms.update(ascii_tokens(tag))
        return frozenset(terms)

    def rerank(self, state: SessionState, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return candidates
        wanted = self._profile_terms(state)
        if not wanted:
            return candidates

        ranked: list[tuple[float, int, Candidate]] = []
        for original_rank, candidate in enumerate(candidates, start=1):
            overlap = len(wanted & self._product_terms(candidate.asin)) / len(wanted)
            bonus = self.weight * overlap / 61
            final = candidate.score + bonus
            ranked.append((
                final,
                original_rank,
                Candidate(candidate.asin, final, {
                    **candidate.components,
                    "profile_tag_overlap": overlap,
                    "profile_rank_bonus": bonus,
                }),
            ))

        ranked.sort(key=lambda item: (-item[0], item[1], item[2].asin))
        return [candidate for _score, _rank, candidate in ranked]
