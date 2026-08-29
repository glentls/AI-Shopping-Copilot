from __future__ import annotations

from starter.personalization.profile_signals import ProfileSignals, parse_profile

DEFAULT_TAIL = ("category", "use_case", "budget", "size", "color", "material", "style", "brand", "feature")


def profile_adjusted_priority(
    base: tuple[str, ...] | list[str],
    profile: dict | ProfileSignals | None,
) -> tuple[str, ...]:
    signals = profile if isinstance(profile, ProfileSignals) else parse_profile(profile if isinstance(profile, dict) else None)
    ordered: list[str] = []
    seen: set[str] = set()

    for attribute in signals.preferred_attributes:
        if attribute not in seen:
            ordered.append(attribute)
            seen.add(attribute)

    for attribute in base:
        if attribute not in seen:
            ordered.append(attribute)
            seen.add(attribute)

    for attribute in DEFAULT_TAIL:
        if attribute not in seen:
            ordered.append(attribute)
            seen.add(attribute)

    return tuple(ordered)
