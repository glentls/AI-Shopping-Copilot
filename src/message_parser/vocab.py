"""Static vocab: word lists, regexes, phrase patterns. Terms were checked
against data/catalog.jsonl and kept only at >=100 occurrences."""

from __future__ import annotations

import re

ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case",
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "just", "me", "my", "of", "on", "or", "please",
    "some", "still", "that", "the", "this", "to", "want", "with", "would",
    "you", "looking", "need", "im", "am", "exploring", "something",
}

MATERIALS = (
    "cotton", "polyester", "poly", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "fabric", "denim", "cashmere", "linen", "suede",
    "canvas", "mesh", "velvet", "satin", "chiffon", "lace", "faux fur",
    "faux leather", "genuine leather", "vegan leather", "patent leather",
    "fur", "fleece", "corduroy", "knit", "elastane", "viscose", "acrylic",
    "bamboo", "microfiber", "twill", "jersey", "terry", "flannel",
    "nubuck", "sherpa", "modal", "tulle", "sequin", "sheepskin",
    "neoprene", "ripstop",
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(re.escape(m) for m in MATERIALS) + r")\b", re.I)

COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "navy", "beige", "tan", "gold",
    "silver", "cream", "maroon", "teal", "turquoise", "khaki", "ivory",
    "burgundy", "coral", "lavender", "mint", "multicolor", "multi-color",
    "rose gold", "sand", "stone", "aqua", "wine", "copper", "sapphire",
    "plum", "charcoal", "olive", "emerald", "bronze", "camel", "peach",
)
COLOR_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in COLORS) + r")\b", re.I)

SIZE_NUMERIC_RE = re.compile(r"\bsize[:\s]*(\d{1,2}(?:\.\d)?)\b", re.I)
SIZE_LETTER_RE = re.compile(r"\bsize[:\s]*(x{0,3}s|x{0,3}l|m)\b", re.I)
SIZE_BARE_LETTER_RE = re.compile(r"\b(xxs|xs|small|medium|large|xl|xxl|xxxl)\b", re.I)
SIZE_WIDTH_RE = re.compile(r"\b(wide|narrow|regular)\s*width\b", re.I)

BUDGET_RE = re.compile(
    r"(?:under|below|less than|no more than|around|about|budget(?: of| around)?)?\s*\$\s?(\d+(?:\.\d{1,2})?)",
    re.I,
)
BUDGET_WORDS_RE = re.compile(
    r"(?:under|below|less than|no more than|around|about)\s+(\d+(?:\.\d{1,2})?)\s*dollars\b",
    re.I,
)

STYLE_KEYWORDS = (
    "formal", "casual", "vintage", "classic", "modern", "sporty", "elegant",
    "slim fit", "slim", "relaxed fit", "relaxed", "oversized", "cropped",
    "high-waisted", "high waisted", "sleeveless", "long sleeve",
    "short sleeve", "crew neck", "v-neck", "button-up", "button up",
    "zip-up", "zip up", "loose fit", "loose", "fitted", "straight leg",
    "skinny", "bootcut", "unisex", "chic", "retro", "plus size", "boho",
    "bohemian", "minimalist", "petite", "maternity", "streetwear",
)
USE_CASE_KEYWORDS = (
    "running", "hiking", "gym", "workout", "yoga", "winter", "summer",
    "outdoor", "work", "office", "wedding", "party", "formal event",
    "everyday", "casual wear", "travel", "beach", "school", "sport",
    "athletic", "training", "walking", "swimming", "cycling", "vacation",
    "sleepwear", "loungewear", "tennis", "camping", "fishing", "golf",
    "basketball", "climbing", "skiing", "hunting", "soccer", "football",
    "boating", "snowboarding", "pilates",
)

OVERRIDE_PATTERNS = (
    "actually", "instead", "ignore my earlier", "ignore that", "scratch that",
    "change of mind", "changed my mind", "on second thought", "never mind that",
    "forget what i said", "rather than that", "let's go with", "lets go with",
)
NO_PREFERENCE_PATTERNS = (
    "don't have a preference", "do not have a preference", "no preference",
    "doesn't matter", "does not matter", "any is fine", "either is fine",
    "either works", "use your judgment", "use your judgement", "up to you",
    "i don't know", "i do not know", "no strong preference", "not picky",
    "whatever works", "you decide", "no particular",
)
VAGUE_PATTERNS = (
    "still exploring", "just looking", "just browsing", "not sure yet",
    "browsing", "open to anything", "not sure what", "no idea yet",
    # Evaluator's generic reprompt when the last turn set no ask_attribute.
    "not quite right yet", "one specific attribute",
)

EXCLUDED_CATEGORY_TERMS = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
    "shoes & jewelry",
}

# A handful of real catalog store/category names are plain short English
# words (e.g. "Key", "Not") and false-positive on unrelated sentences.
GENERIC_SINGLE_WORD_BLOCKLIST = {
    "key", "not", "so", "up", "in", "on", "at", "by", "or", "all", "new",
    "one", "top", "set", "box", "plus", "its", "out", "off", "non", "our",
    "any", "get", "buy", "shop", "store", "and", "for", "with", "you",
    "your", "what", "ask", "about", "right", "yet", "quite", "those",
    "options",
}
MIN_SINGLE_WORD_VOCAB_LEN = 4

# Customers commonly type these as one merged word; catalog categories store
# them space/hyphen-separated. Expanded into their constituent tokens before
# vocab n-gram matching (see parser._extract_attributes).
COMPOUND_ALIASES = {
    "tshirt": "t shirts",
    "tshirts": "t shirts",
    "flipflop": "flip flops",
    "flipflops": "flip flops",
    "gstring": "g strings",
    "gstrings": "g strings",
    "coverup": "cover ups",
    "coverups": "cover ups",
    "buttondown": "button down",
    "crossbody": "cross body",
    "carryon": "carry ons",
    "carryons": "carry ons",
}
