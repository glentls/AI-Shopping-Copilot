"""Word lists and synonym maps, one canonical value per concept.

LANE A OWNS THIS FILE. This is a deliberately thin starting point -- it exists
so the skeleton runs end to end. Expanding it is the bulk of Lane A's work.

Every pattern is matched with word boundaries. A naive substring search for
"red" hits 9,348 products in this catalog; \\bred\\b hits 1,916. The rest are
embroidered, zippered, reduce, inspired, covered, credit, textured, layered.
"""

from __future__ import annotations

import re

# slot -> {canonical value: [surface forms]}
LEXICON: dict[str, dict[str, list[str]]] = {
    "material": {
        "cotton": ["cotton"],
        "polyester": ["polyester"],
        "leather": ["leather", "genuine leather", "full grain leather"],
        "nylon": ["nylon"],
        "wool": ["wool", "merino"],
        "spandex": ["spandex", "elastane", "lycra"],
        "silk": ["silk"],
        "denim": ["denim"],
        "fleece": ["fleece"],
        "rubber": ["rubber"],
        "stainless steel": ["stainless steel"],
        "sterling silver": ["sterling silver"],
    },
    "color": {
        "black": ["black"],
        "white": ["white"],
        "blue": ["blue", "navy"],
        "red": ["red"],
        "pink": ["pink"],
        "green": ["green"],
        "brown": ["brown", "tan"],
        "gray": ["gray", "grey"],
        "purple": ["purple"],
        "yellow": ["yellow"],
        "beige": ["beige", "ivory", "cream"],
        "gold": ["gold"],
        "silver": ["silver"],
    },
    "use_case": {
        "travel": ["travel", "traveling", "travelling", "vacation", "trip"],
        "hiking": ["hiking", "trail", "trekking"],
        "running": ["running", "jogging"],
        "walking": ["walking", "long walks"],
        "gym": ["gym", "workout", "training", "fitness"],
        "work": ["work", "office", "business"],
        "casual": ["casual", "everyday", "daily"],
        "formal": ["formal", "wedding", "party", "dressy"],
        "outdoor": ["outdoor", "camping"],
        "winter": ["winter", "cold weather", "snow"],
        "summer": ["summer", "beach"],
        "sleep": ["sleep", "sleeping", "pajama", "lounge"],
    },
    "feature": {
        "waterproof": ["waterproof", "water resistant", "water-resistant"],
        "breathable": ["breathable", "ventilated"],
        "moisture wicking": ["moisture wicking", "moisture-wicking", "quick dry", "quick-dry"],
        "insulated": ["insulated", "thermal", "fleece lined"],
        "non slip": ["non slip", "non-slip", "anti slip", "slip resistant"],
        "lightweight": ["lightweight", "light weight"],
        "comfortable": ["comfortable", "comfort", "cushioned", "arch support"],
        "adjustable": ["adjustable"],
        "pockets": ["pockets", "pocket"],
        "stretch": ["stretch", "stretchy", "elastic"],
    },
    "style": {
        "long sleeve": ["long sleeve", "long-sleeve"],
        "short sleeve": ["short sleeve", "short-sleeve"],
        "sleeveless": ["sleeveless", "tank"],
        "v neck": ["v neck", "v-neck"],
        "crew neck": ["crew neck", "crewneck"],
        "hooded": ["hooded", "hoodie"],
        "loose": ["loose", "relaxed fit", "oversized"],
        "slim": ["slim fit", "skinny", "fitted"],
        "high waisted": ["high waisted", "high-waisted"],
        "pullover": ["pullover"],
        "zip up": ["zip up", "zip-up", "full zip"],
    },
    "size": {
        "plus": ["plus size", "plus-size"],
        "petite": ["petite"],
        "wide": ["wide width", "wide fit"],
        "narrow": ["narrow width"],
        "tall": ["big and tall", "tall"],
    },
}

# Phrases that mean the customer is replacing an earlier preference.
OVERRIDE_CUES = (
    "actually", "instead", "on second thought", "ignore what i said",
    "ignore my earlier", "changed my mind", "scratch that", "forget what i said",
    "rather than", "no longer",
)

# Phrases that mean the customer has no preference for what we just asked.
NO_PREFERENCE_CUES = (
    "no preference", "don't have a preference", "dont have a preference",
    "you decide", "use your judgment", "use your judgement", "doesn't matter",
    "doesnt matter", "either is fine", "no strong feelings", "up to you",
    "i'm not fussy", "no idea",
)

# Words that flip the polarity of the value that follows them.
NEGATION_CUES = ("not", "no", "without", "avoid", "anything but", "except", "don't want", "dont want")


def compile_patterns() -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    """slot -> [(canonical value, word-bounded pattern)]."""
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for slot, values in LEXICON.items():
        entries: list[tuple[str, re.Pattern[str]]] = []
        for canonical, surfaces in values.items():
            alternation = "|".join(re.escape(s) for s in sorted(surfaces, key=len, reverse=True))
            entries.append((canonical, re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)))
        compiled[slot] = entries
    return compiled


PATTERNS = compile_patterns()
