"""Intent Router component.

Classifies a raw customer message into one of four scenarios
(``buying`` / ``browsing`` / ``intent_override`` / ``boundary``) and extracts
structured attributes, delegating the linguistic work to ``src.message_parser``.

Public API:
    detect_scenario(message, history) -> str
    extract_attributes(message)       -> dict
"""

from src.intent_router.router import detect_scenario, extract_attributes, parse_message

__all__ = ["detect_scenario", "extract_attributes", "parse_message"]
