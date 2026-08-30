"""A realistic shopper simulator, as an alternative to the evaluator's templates.

The official simulator in ``evaluator.local_evaluator`` speaks six fixed
sentences ("A key requirement is: X.", "For that, what matters is: X."). That is
fine for scoring but it means the public set cannot tell you whether the agent
understands *shoppers* or merely those six strings. The competition spec warns
the organizer may paraphrase, so the difference matters.

This module builds an equivalent customer from the same ground-truth product,
but phrases turns the way people actually shop. The design follows patterns
reported for multi-turn shopping benchmarks:

* mission types - goal-directed "find a specific solution", open-ended
  "explore and discover", and "conversational navigation" where intent moves
  mid-session (Shopping Reasoning Bench, arXiv:2606.12608);
* real shoppers search by style, occasion, colour, fit and inspiration rather
  than product names, and gift buyers narrow by recipient, occasion, budget and
  sentiment (MMShopBench, arXiv:2607.29002);
* a simulated user holds a private preference set and discloses it
  incrementally in response to clarification (UserSimCRS,
  github.com/iai-group/UserSimCRS).

Every fact the simulated customer states is derived from the real target
product, so a competent agent can still converge. Only the wording changes.
"""

from __future__ import annotations

import random
import re
from typing import Any


# =============================================================================
# VOCABULARY
# =============================================================================

# Deliberately wider than starter/extractor.py, which is missing most of these.
COLOR_WORDS = (
    "black", "white", "blue", "navy", "royal blue", "sky blue", "red", "wine red",
    "pink", "hot pink", "blush", "green", "olive", "mint", "brown", "tan", "camel",
    "beige", "khaki", "cream", "ivory", "gray", "grey", "charcoal", "purple",
    "lavender", "violet", "yellow", "mustard", "orange", "coral", "burgundy",
    "maroon", "teal", "turquoise", "gold", "silver", "rose gold", "bronze",
    "multicolor", "clear",
)

MATERIAL_WORDS = (
    "cotton", "organic cotton", "polyester", "nylon", "leather", "faux leather",
    "genuine leather", "patent leather", "suede", "wool", "merino wool", "cashmere",
    "spandex", "elastane", "silk", "satin", "rayon", "viscose", "modal", "linen",
    "denim", "canvas", "mesh", "fleece", "sherpa", "knit", "ribbed knit", "velvet",
    "chiffon", "corduroy", "flannel", "twill", "jersey", "bamboo", "microfiber",
    "neoprene", "rubber", "eva", "alloy", "zinc alloy", "sterling silver",
    "stainless steel", "titanium", "brass", "copper", "resin", "acrylic",
    "cubic zirconia", "rhinestone", "pearl", "shell", "wood", "recycled polyester",
)

PATTERN_WORDS = (
    "striped", "stripe", "floral", "plaid", "checkered", "gingham", "polka dot",
    "houndstooth", "camo", "camouflage", "leopard", "animal print", "tie dye",
    "geometric", "paisley", "colorblock", "graphic print", "embroidered",
    "sequin", "glitter", "marble", "abstract", "solid",
)

USE_CASE_WORDS = (
    "running", "hiking", "walking", "gym", "yoga", "pilates", "training",
    "crossfit", "weightlifting", "travel", "work", "office", "nursing",
    "construction", "wedding", "bridesmaid", "prom", "party", "cocktail",
    "festival", "beach", "poolside", "winter", "summer", "spring", "fall",
    "rain", "snow", "camping", "fishing", "cycling", "skiing", "snowboarding",
    "golf", "tennis", "basketball", "soccer", "boxing", "hunting", "climbing",
    "commuting", "lounging", "sleeping", "everyday", "casual", "formal",
    "business casual", "school", "graduation", "maternity", "postpartum",
    "date night", "vacation", "hot weather", "cold weather",
)

STYLE_WORDS = (
    "vintage", "retro", "y2k", "minimalist", "bohemian", "boho", "classic",
    "modern", "elegant", "sporty", "athleisure", "streetwear", "preppy",
    "gothic", "punk", "western", "cowboy", "nautical", "coastal", "cottagecore",
    "oversized", "slim fit", "regular fit", "relaxed fit", "loose fit",
    "skinny", "bootcut", "flared", "wide leg", "tapered", "high waisted",
    "mid rise", "low rise", "cropped", "longline", "tunic", "bodycon",
    "a-line", "wrap", "pleated", "ruched", "asymmetric", "layered",
    "sleeveless", "long sleeve", "short sleeve", "three quarter sleeve",
    "puff sleeve", "bell sleeve", "v-neck", "crew neck", "scoop neck",
    "turtleneck", "mock neck", "off shoulder", "halter", "strapless",
    "hooded", "collared", "button down", "open front", "zip up", "pullover",
)

FEATURE_WORDS = (
    "waterproof", "water resistant", "windproof", "quick dry", "moisture wicking",
    "breathable", "insulated", "thermal", "fleece lined", "sherpa lined",
    "lightweight", "heavy duty", "non slip", "slip resistant", "anti slip",
    "arch support", "cushioned", "memory foam", "orthotic", "wide toe box",
    "steel toe", "adjustable", "elastic waist", "drawstring", "stretchy",
    "four way stretch", "wrinkle free", "machine washable", "hand wash",
    "pockets", "zipper pockets", "hidden pocket", "rfid blocking",
    "hypoallergenic", "nickel free", "tarnish resistant", "reversible",
    "packable", "foldable", "uv protection", "spf 50", "reflective",
    "pull on", "lace up", "velcro", "magnetic closure", "buckle closure",
    "snap closure", "hook and eye", "padded", "underwire", "wire free",
    "seamless", "tagless", "gift box", "adjustable strap", "removable strap",
)

# Detail keys worth reading verbatim, mapped onto contract attributes.
DETAIL_ATTRIBUTE_KEYS = {
    "color": "color",
    "colour": "color",
    "material": "material",
    "fabric type": "material",
    "outer material": "material",
    "material composition": "material",
    "style": "style",
    "pattern": "style",
    "shape": "style",
    "size": "size",
    "fit type": "size",
    "age range (description)": "use_case",
    "suggested users": "use_case",
    "sport type": "use_case",
    "occasion": "use_case",
    "season": "use_case",
    "closure type": "feature",
    "special feature": "feature",
    "sole material": "feature",
    "heel type": "feature",
    "care instructions": "feature",
    "water resistance level": "feature",
    "neck style": "style",
    "sleeve type": "style",
}

DEPARTMENTS = {
    "womens": "women's", "women": "women's", "female": "women's",
    "mens": "men's", "men": "men's", "male": "men's",
    "girls": "girls'", "boys": "boys'", "unisex": "unisex", "baby": "baby",
    "unisex-adult": "unisex", "unisex-child": "unisex",
}

GIFT_RECIPIENTS = (
    "wife", "husband", "sister", "brother", "mom", "dad", "daughter", "son",
    "girlfriend", "boyfriend", "best friend", "niece", "nephew", "coworker",
    "grandmother", "grandfather", "roommate", "teacher",
)

GIFT_OCCASIONS = (
    "birthday", "anniversary", "Christmas", "graduation", "Mother's Day",
    "Father's Day", "wedding", "housewarming", "Valentine's Day",
    "baby shower", "retirement", "promotion",
)

SIZE_RE = re.compile(
    r"\b(?:size\s+)?(xx?s|xx?l|xxxl|small|medium|large|petite|plus size|tall|"
    r"(?:us\s*)?(?:[4-9]|1[0-5])(?:\.5)?\s*(?:wide|narrow|regular)?)\b",
    re.IGNORECASE,
)

MISSION_FOR_SCENARIO = {
    "buying": "find_specific_solution",
    "browsing": "explore_and_discover",
    "intent_override": "conversational_navigation",
    "boundary": "explore_and_discover",
}

# Attributes the customer will volunteer up front, best signal first.
DISCLOSURE_ORDER = ("color", "material", "use_case", "budget", "style", "brand", "size", "feature")

JUNK_DETAIL_VALUES = {
    "", "n/a", "na", "none", "no", "yes", "-", "other", "unknown", "solid",
    "not applicable", "see description",
}


# =============================================================================
# FACET EXTRACTION
# =============================================================================


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items() if item not in (None, "", []))
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def _corpus(product: dict) -> str:
    """Only fields that describe *this* variant.

    ``description`` is excluded on purpose: it routinely lists colours and
    materials of sibling variants, and a customer stating an untrue constraint
    would make the session unwinnable rather than merely hard.
    """
    return " ".join(
        _text(product.get(field))
        for field in ("title", "features", "details", "categories")
    )


def _first_present(words: tuple[str, ...], haystack: str) -> str | None:
    """Earliest match wins; longer match breaks a tie at the same position.

    Order matters. In "07-navy+cream" the primary colour is navy, and in
    "faux leather" the material is faux leather, not leather - both fall out of
    earliest-start with a longest-match tiebreak.
    """
    lowered = haystack.lower()
    best: tuple[int, int, str] | None = None
    for word in words:
        match = re.search(rf"\b{re.escape(word)}\b", lowered)
        if not match:
            continue
        candidate = (match.start(), -len(word), word)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best else None


def _all_present(words: tuple[str, ...], haystack: str, limit: int = 3) -> list[str]:
    lowered = haystack.lower()
    found: list[str] = []
    for word in sorted(words, key=len, reverse=True):
        if len(found) >= limit:
            break
        if any(word in existing for existing in found):
            continue
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            found.append(word)
    return found


def _clean_detail(value: Any, limit: int = 40) -> str | None:
    text = re.sub(r"\s+", " ", str(value)).strip(" -;,.")
    if not text or text.lower() in JUNK_DETAIL_VALUES or len(text) > limit:
        return None
    if re.fullmatch(r"[\d.,\s]+", text):
        return None
    return text


def _details(product: dict) -> dict[str, str]:
    details = product.get("details")
    return {str(key).lower(): value for key, value in details.items()} if isinstance(details, dict) else {}


def _normalize_detail(attribute: str, value: str) -> str | None:
    """Catalog colour/material fields carry SKU noise like "04-black+beige"."""
    vocabulary = {"color": COLOR_WORDS, "material": MATERIAL_WORDS}.get(attribute)
    if vocabulary:
        match = _first_present(vocabulary, value)
        if match:
            return match
        if not re.fullmatch(r"[a-z][a-z /-]{1,24}", value.lower()):
            return None
        return value.lower()
    return value


def _detail_facets(product: dict) -> dict[str, str]:
    """Read the catalog's own structured attributes before guessing from prose."""
    found: dict[str, str] = {}
    for key, value in _details(product).items():
        attribute = DETAIL_ATTRIBUTE_KEYS.get(key)
        if not attribute or attribute in found:
            continue
        cleaned = _clean_detail(value)
        if not cleaned:
            continue
        normalized = _normalize_detail(attribute, cleaned)
        if normalized:
            found[attribute] = normalized
    return found


def _department(product: dict) -> str | None:
    for key, value in _details(product).items():
        if "department" in key:
            return DEPARTMENTS.get(str(value).strip().lower())
    return None


def feature_pool(product: dict, limit: int = 60, count: int = 4) -> list[str]:
    """Several short, concrete specs. Shoppers cite these one at a time."""
    store = _text(product.get("store")).strip()
    candidates: list[str] = []

    corpus = _corpus(product)
    candidates.extend(_all_present(FEATURE_WORDS, corpus, limit=3))

    details = _details(product)
    for key in ("closure type", "special feature", "sole material", "heel type", "care instructions"):
        cleaned = _clean_detail(details.get(key, ""))
        if cleaned:
            candidates.append(cleaned.lower())

    features = product.get("features")
    if isinstance(features, list):
        for item in features:
            item = re.sub(r"\s+", " ", str(item)).strip(" -;,.")
            lowered = item.lower()
            if not (4 <= len(item) <= limit):
                continue
            if lowered.startswith(("imported", "made in", "brand ", "manufacturer ")):
                continue
            if store and store.lower() in lowered:
                continue
            candidates.append(item)

    unique: list[str] = []
    for item in candidates:
        if any(item.lower() in existing.lower() or existing.lower() in item.lower() for existing in unique):
            continue
        unique.append(item)
        if len(unique) >= count:
            break
    return unique


def _budget_phrase(price: float | None, rng: random.Random) -> str | None:
    if price is None or price <= 0:
        return None
    # Always satisfiable by the real product, so a good agent is never punished.
    style = rng.choice(("under", "around", "between", "max"))
    if style == "under":
        cap = max(10, int(price * 1.2) + 5)
        return f"under ${cap}"
    if style == "max":
        cap = max(10, int(price * 1.15) + 5)
        return f"max ${cap}"
    if style == "around":
        return f"around ${int(round(price))}"
    low = max(5, int(price * 0.7))
    high = int(price * 1.3) + 5
    return f"between ${low} and ${high}"


def extract_facets(product: dict, rng: random.Random) -> dict[str, str]:
    """Typed, human-sounding constraints, all true of this product.

    Structured ``details`` win over prose: the catalog's own ``color`` /
    ``material`` / ``pattern`` / ``closure type`` fields are authoritative,
    where a regex over the title is a guess.
    """
    corpus = _corpus(product)
    title = _text(product.get("title"))
    department = _department(product)
    facets: dict[str, str] = dict(_detail_facets(product))

    if "color" not in facets:
        color = _first_present(COLOR_WORDS, title) or _first_present(COLOR_WORDS, corpus)
        if color:
            facets["color"] = color

    if "material" not in facets:
        material = _first_present(MATERIAL_WORDS, corpus)
        if material:
            facets["material"] = material

    if "use_case" not in facets:
        use_case = _first_present(USE_CASE_WORDS, corpus)
        if use_case:
            facets["use_case"] = use_case

    if "style" not in facets:
        style = _first_present(STYLE_WORDS, corpus) or _first_present(PATTERN_WORDS, corpus)
        if style and department:
            facets["style"] = f"{department} {style}"
        elif style:
            facets["style"] = style
        elif department:
            facets["style"] = department

    store = _text(product.get("store")).strip()
    if store and len(store) <= 40:
        facets["brand"] = store

    budget = _budget_phrase(
        product.get("price") if isinstance(product.get("price"), (int, float)) else None, rng
    )
    if budget:
        facets["budget"] = budget

    if "size" not in facets:
        size_match = SIZE_RE.search(title)
        if size_match:
            facets["size"] = f"size {size_match.group(1).lower()}"
        elif department and department not in facets.get("style", ""):
            facets["size"] = f"{department} fit"

    if "feature" not in facets:
        pool = feature_pool(product)
        if pool:
            facets["feature"] = pool[0]

    return facets


# =============================================================================
# PERSONA
# =============================================================================


def build_persona(rng: random.Random, difficulty: str) -> dict:
    gift = rng.random() < (0.30 if difficulty != "hard" else 0.20)
    return {
        "shopping_for": "gift" if gift else "self",
        "recipient": rng.choice(GIFT_RECIPIENTS) if gift else None,
        "occasion": rng.choice(GIFT_OCCASIONS) if gift else None,
        "tone": rng.choice(("brisk", "chatty", "hesitant")),
        "price_sensitive": rng.random() < 0.45,
    }


# =============================================================================
# TEMPLATE BANKS
# =============================================================================

# Verb variety on purpose. starter/extractor.py only matches
# "looking for", "shopping for", "searching for", "i need", "i want" - the
# others are the paraphrase-robustness probe.
OPEN_SPECIFIC = (
    "I'm looking for {category}. {constraints}.",
    "I need {category} - {constraints}.",
    "Shopping for {category}, and {constraints}.",
    "I want {category}. Must be {constraints}.",
    "I'm after {category}, ideally {constraints}.",
    "Trying to find {category} that is {constraints}.",
    "Hunting for {category} - {constraints} if possible.",
    "Need {category}. {constraints} is the main thing.",
)

OPEN_EXPLORE = (
    "I'm looking for {category}, but I'm still exploring.",
    "Just browsing {category} for now.",
    "Show me some {category} - I haven't decided on the details yet.",
    "I'm thinking about {category}, not sure what I want exactly.",
    "Been meaning to get {category}. Open to suggestions.",
    "What {category} would you recommend? I'm flexible.",
    "I'm browsing {category}. Nothing specific in mind.",
)

OPEN_REPLACEMENT = (
    "My old {category} finally wore out, so I need a replacement.",
    "I had {category} I loved and they're falling apart now.",
    "Time to replace my {category}. Something similar would be great.",
    "Looking to replace {category} I've had for years.",
)

OPEN_GIFT = (
    "I'm looking for a gift for my {recipient} - {category}, for their {occasion}.",
    "Need a {occasion} gift for my {recipient}. Thinking {category}.",
    "Getting {category} as a {occasion} present for my {recipient}.",
    "My {recipient}'s {occasion} is coming up. I want to get them {category}.",
)

OPEN_VAGUE = (
    "I need something for {use_case}, not sure what exactly.",
    "Help me find something nice, maybe {category}?",
    "I don't really know what I'm after. Something in {category} I guess.",
    "Looking for ideas around {category}.",
)

REPLY_CONFIDENT = (
    "Yes - {values}.",
    "{values}, definitely.",
    "It has to be {values}.",
    "I'd want {values}.",
    "Go with {values}.",
)

REPLY_SOFT = (
    "I'd say {values}, but I'm a bit flexible.",
    "Probably {values}, if that helps.",
    "Leaning towards {values}.",
    "Something like {values} would work.",
    "Ideally {values}, though I could compromise.",
)

REPLY_HESITANT = (
    "Hmm, maybe {values}? Not sure it matters much.",
    "I guess {values}, if I had to pick.",
    "Possibly {values}. Hard to say.",
    "Something along the lines of {values}, I think.",
)

REPLY_NOTHING = (
    "No strong feelings on {attribute}.",
    "I don't have an additional preference for {attribute}.",
    "{attribute} isn't something I've thought about.",
    "Nothing specific for {attribute}.",
)

REPLY_BOUNDARY = (
    "I don't have a preference for {attribute}; please use your judgment.",
    "No preference on {attribute} - you pick.",
    "Honestly, {attribute} is up to you.",
    "I really don't mind about {attribute}. Your call.",
)

REPLY_NO_QUESTION = (
    "Those options are not quite right yet. Ask me about one specific attribute.",
    "None of those work. What else do you need to know?",
    "Not quite. Ask me something specific and I'll answer.",
    "Still not right - what would help you narrow it down?",
)

OVERRIDE_MESSAGES = (
    "Actually, ignore my earlier preference. What I need is: {new_value}.",
    "Actually, scratch that - {new_value} is what I really need.",
    "Change of plan. Forget what I said before; I need {new_value} instead.",
    "Hold on - ignore that. The important thing is {new_value}.",
    "I've changed my mind. Instead of that, I need {new_value}.",
)

# Attributes worth volunteering per difficulty, and how many at turn 1.
OPENING_CONSTRAINT_COUNT = {"easy": 2, "medium": 1, "hard": 0}


def _join(values: list[str]) -> str:
    if len(values) <= 1:
        return values[0] if values else ""
    return "; ".join(values)


# =============================================================================
# CUSTOMER
# =============================================================================


class RealisticCustomer:
    """Stateful shopper. Same contract as the evaluator's customer policy."""

    def __init__(
        self,
        sample: dict,
        product: dict,
        coarse_category: str,
        max_turns: int = 10,
    ) -> None:
        self.sample_id = str(sample["sample_id"])
        self.scenario = str(sample["scenario_type"])
        self.difficulty = str(sample.get("difficulty_bucket") or "medium")
        self.mission = MISSION_FOR_SCENARIO.get(self.scenario, "explore_and_discover")
        self.category = coarse_category
        self.rng = random.Random(f"{self.sample_id}\0{self.scenario}\0{self.difficulty}")
        self.facets = extract_facets(product, self.rng)
        # Spare specs, so a second "what feature matters?" gets a new answer
        # instead of "no preference".
        primary = self.facets.get("feature", "")
        self.spare_features = [
            item for item in feature_pool(product)
            if item.lower() != primary.lower()
        ]
        self.persona = sample.get("persona") or build_persona(self.rng, self.difficulty)
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_turn = self.rng.choice((3, 4))
        self._override_value: str | None = None

    # ---------------------------------------------------------------------

    def _pending(self, attribute: str) -> str | None:
        if attribute in self.disclosed:
            return None
        return self.facets.get(attribute)

    def _best_pending(self, count: int) -> list[str]:
        picked: list[str] = []
        for attribute in DISCLOSURE_ORDER:
            if len(picked) >= count:
                break
            value = self._pending(attribute)
            if value:
                self.disclosed.add(attribute)
                picked.append(value)
        return picked

    # ---------------------------------------------------------------------

    def opening(self) -> str:
        persona = self.persona
        count = OPENING_CONSTRAINT_COUNT.get(self.difficulty, 1)

        if self.scenario == "intent_override":
            # Open with a real but *secondary* preference, then reverse it later.
            decoy = None
            for attribute in ("style", "color", "material", "feature"):
                decoy = self._pending(attribute)
                if decoy:
                    self.disclosed.add(attribute)
                    break
            opener = self.rng.choice(OPEN_SPECIFIC)
            return opener.format(category=self.category, constraints=decoy or "something classic")

        if persona.get("shopping_for") == "gift" and self.scenario != "boundary":
            template = self.rng.choice(OPEN_GIFT)
            text = template.format(
                category=self.category,
                recipient=persona.get("recipient") or "friend",
                occasion=persona.get("occasion") or "birthday",
            )
            extras = self._best_pending(count)
            return f"{text} {self.rng.choice(('Budget-wise', 'Ideally'))}: {_join(extras)}." if extras else text

        if self.scenario == "buying":
            constraints = self._best_pending(max(1, count))
            if not constraints:
                return self.rng.choice(OPEN_REPLACEMENT).format(category=self.category)
            return self.rng.choice(OPEN_SPECIFIC).format(
                category=self.category, constraints=_join(constraints)
            )

        # browsing / boundary
        if self.difficulty == "hard":
            template = self.rng.choice(OPEN_VAGUE)
            return template.format(
                category=self.category,
                use_case=self.facets.get("use_case", "everyday use"),
            )
        text = self.rng.choice(OPEN_EXPLORE).format(category=self.category)
        extras = self._best_pending(count)
        return f"{text} {_join(extras)}." if extras else text

    # ---------------------------------------------------------------------

    def override_message(self) -> tuple[str, str]:
        """Returns (message, new_value). Called once, at ``override_turn``."""
        new_value = None
        for attribute in ("material", "color", "use_case", "feature", "style"):
            new_value = self._pending(attribute)
            if new_value:
                self.disclosed.add(attribute)
                break
        new_value = new_value or self.category
        self._override_value = new_value
        return self.rng.choice(OVERRIDE_MESSAGES).format(new_value=new_value), new_value

    # ---------------------------------------------------------------------

    def reply(self, ask_attribute: object) -> str:
        attribute = ask_attribute if isinstance(ask_attribute, str) else None

        if self.scenario == "boundary" and not self.boundary_used and attribute:
            self.boundary_used = True
            return self.rng.choice(REPLY_BOUNDARY).format(attribute=attribute)

        if not attribute:
            return self.rng.choice(REPLY_NO_QUESTION)

        if attribute == "other" or attribute == "category":
            # Wildcard: hand over whatever is still undisclosed, best first.
            values = self._best_pending(2)
        else:
            value = self._pending(attribute)
            values = []
            if value:
                self.disclosed.add(attribute)
                values.append(value)

        if not values and attribute in ("feature", "other", "style") and self.spare_features:
            values = [self.spare_features.pop(0)]

        if not values:
            return self.rng.choice(REPLY_NOTHING).format(attribute=attribute)

        if self.difficulty == "easy":
            bank = REPLY_CONFIDENT
        elif self.difficulty == "medium":
            bank = REPLY_SOFT
        else:
            bank = REPLY_HESITANT
        if self.persona.get("tone") == "hesitant":
            bank = REPLY_HESITANT
        return self.rng.choice(bank).format(values=_join(values))
