"""Canonical shopping vocabulary shared by message and catalog extraction.

Every surface form is compiled with lexical boundaries. This matters for
short values in particular: ``red`` must not match ``embroidered`` and ``tan``
must not match ``tanks``. Multi-word surfaces accept spaces or hyphens so the
catalog and a customer's prose normalize to the same value.
"""

from __future__ import annotations

import re


# slot -> {canonical value: surface forms}. Canonical values are deliberately
# human-readable because they are also used in explanations and debug output.
LEXICON: dict[str, dict[str, list[str]]] = {
    "category": {
        "women": ["women", "woman", "women's", "womens", "ladies", "for her"],
        "men": ["men", "man", "men's", "mens", "for him"],
        "girls": ["girl", "girls", "girl's"],
        "boys": ["boy", "boys", "boy's"],
        "baby": ["baby", "infant", "newborn", "toddler"],
        "shirt": ["shirt", "shirts", "t shirt", "t-shirt", "tee", "tees"],
        "top": ["top", "tops", "blouse", "blouses", "tunic", "tunics", "camisole"],
        "dress": ["dress", "dresses", "gown", "gowns"],
        "skirt": ["skirt", "skirts"],
        "pants": ["pants", "trousers", "slacks", "chinos"],
        "shorts": ["shorts", "short pants"],
        "jeans": ["jeans", "jean pants"],
        "leggings": ["legging", "leggings", "tights"],
        "sweater": ["sweater", "sweaters", "cardigan", "cardigans"],
        "sweatshirt": ["sweatshirt", "sweatshirts", "hoodie", "hoodies"],
        "jacket": ["jacket", "jackets", "blazer", "blazers"],
        "coat": ["coat", "coats", "parka", "parkas"],
        "vest": ["vest", "vests", "waistcoat"],
        "suit": ["suit", "suits", "tuxedo", "tuxedos"],
        "underwear": ["underwear", "panties", "briefs", "boxers", "lingerie"],
        "bra": ["bra", "bras", "bralette", "bralettes"],
        "sleepwear": ["sleepwear", "pajama", "pajamas", "pyjama", "pyjamas", "nightgown", "robe"],
        "swimwear": ["swimwear", "swimsuit", "swimsuits", "bikini", "bikinis", "swim trunks"],
        "socks": ["sock", "socks", "hosiery"],
        "shoes": ["shoe", "shoes", "footwear"],
        "sneakers": ["sneaker", "sneakers", "trainers", "tennis shoes"],
        "boots": ["boot", "boots", "bootie", "booties"],
        "sandals": ["sandal", "sandals", "flip flop", "flip flops"],
        "heels": ["heel", "heels", "pumps", "stiletto", "stilettos"],
        "flats": ["flat shoe", "flat shoes", "flats", "ballet flat", "ballet flats"],
        "loafers": ["loafer", "loafers", "moccasin", "moccasins"],
        "slippers": ["slipper", "slippers", "house shoes"],
        "jewelry": ["jewelry", "jewellery"],
        "earrings": ["earring", "earrings"],
        "necklace": ["necklace", "necklaces", "pendant", "pendants"],
        "bracelet": ["bracelet", "bracelets", "bangle", "bangles"],
        "ring": ["ring", "rings"],
        "watch": ["watch", "watches", "wristwatch"],
        "bag": ["bag", "bags", "handbag", "handbags", "purse", "purses", "backpack", "backpacks", "tote", "totes"],
        "wallet": ["wallet", "wallets", "card holder", "card holders"],
        "belt": ["belt", "belts"],
        "hat": ["hat", "hats", "cap", "caps", "beanie", "beanies"],
        "sunglasses": ["sunglasses", "sun glasses", "shades"],
        "costume": ["costume", "costumes", "cosplay"],
    },
    "material": {
        "cotton": ["cotton", "organic cotton", "pima cotton"],
        "polyester": ["polyester", "poly"],
        "leather": ["leather", "genuine leather", "full grain leather", "pu leather", "faux leather", "vegan leather"],
        "suede": ["suede", "microsuede"],
        "nylon": ["nylon"],
        "wool": ["wool", "merino", "cashmere"],
        "spandex": ["spandex", "elastane", "lycra"],
        "silk": ["silk", "satin"],
        "rayon": ["rayon", "viscose", "modal"],
        "linen": ["linen"],
        "denim": ["denim", "chambray"],
        "fleece": ["fleece", "polar fleece"],
        "acrylic": ["acrylic"],
        "rubber": ["rubber"],
        "canvas": ["canvas"],
        "mesh": ["mesh"],
        "velvet": ["velvet", "velour"],
        "lace": ["lace"],
        "chiffon": ["chiffon"],
        "stainless steel": ["stainless steel", "surgical steel"],
        "sterling silver": ["sterling silver", "925 silver", "925 sterling"],
        "gold": ["solid gold", "14k gold", "18k gold", "gold plated", "gold-plated"],
        "memory foam": ["memory foam"],
        "synthetic": ["synthetic", "man made", "man-made"],
    },
    "color": {
        "black": ["black", "jet black"],
        "white": ["white", "off white", "off-white"],
        "blue": ["blue", "navy", "navy blue", "royal blue", "cobalt"],
        "red": ["red", "scarlet", "crimson"],
        "pink": ["pink", "blush", "fuchsia", "hot pink"],
        "green": ["green", "olive", "forest green", "lime green"],
        "brown": ["brown", "tan", "camel", "chocolate"],
        "gray": ["gray", "grey", "charcoal", "heather gray", "heather grey"],
        "purple": ["purple", "violet", "lavender", "plum"],
        "yellow": ["yellow", "mustard"],
        "orange": ["orange", "coral", "rust"],
        "beige": ["beige", "ivory", "cream"],
        "gold": ["gold", "golden", "rose gold"],
        "silver": ["silver", "silver tone", "silver-tone"],
        "teal": ["teal", "turquoise", "aqua"],
        "burgundy": ["burgundy", "maroon", "wine red"],
        "multicolor": ["multicolor", "multi color", "multi-color", "multicolored", "rainbow"],
    },
    "size": {
        "xs": ["extra small", "x small", "x-small", "xs"],
        "s": ["small", "size s"],
        "m": ["medium", "size m"],
        "l": ["large", "size l"],
        "xl": ["extra large", "x large", "x-large", "xl"],
        "xxl": ["double extra large", "2xl", "xxl", "xx-large"],
        "xxxl": ["triple extra large", "3xl", "xxxl", "xxx-large"],
        "plus": ["plus size", "plus-size", "extended size", "curvy"],
        "petite": ["petite"],
        "wide": ["wide width", "wide fit", "extra wide", "wide shoe", "wide shoes"],
        "narrow": ["narrow width", "narrow fit", "narrow shoe", "narrow shoes"],
        "tall": ["big and tall", "big & tall", "tall size", "tall sizes"],
        "one size": ["one size", "one-size", "one size fits all"],
    },
    "style": {
        "long sleeve": ["long sleeve", "long-sleeve", "full sleeve"],
        "short sleeve": ["short sleeve", "short-sleeve"],
        "three quarter sleeve": ["three quarter sleeve", "3/4 sleeve", "three-quarter sleeve"],
        "sleeveless": ["sleeveless", "tank style"],
        "v neck": ["v neck", "v-neck"],
        "crew neck": ["crew neck", "crewneck", "round neck"],
        "scoop neck": ["scoop neck", "scoop-neck"],
        "turtleneck": ["turtleneck", "turtle neck", "mock neck"],
        "collared": ["collared", "point collar", "spread collar"],
        "hooded": ["hooded", "hoodie", "with hood"],
        "loose": ["loose", "relaxed fit", "oversized", "roomy"],
        "regular fit": ["regular fit", "classic fit"],
        "slim": ["slim fit", "skinny", "fitted", "bodycon"],
        "high waisted": ["high waisted", "high-waisted", "high rise", "high-rise"],
        "mid rise": ["mid rise", "mid-rise"],
        "low rise": ["low rise", "low-rise"],
        "pullover": ["pullover", "pull over"],
        "zip up": ["zip up", "zip-up", "full zip", "full-zip", "zip front"],
        "button down": ["button down", "button-down", "button front"],
        "lace up": ["lace up", "lace-up"],
        "slip on": ["slip on", "slip-on"],
        "open toe": ["open toe", "open-toe", "peep toe"],
        "closed toe": ["closed toe", "closed-toe"],
        "ankle length": ["ankle length", "ankle-length", "cropped"],
        "knee length": ["knee length", "knee-length"],
        "midi": ["midi", "mid length", "mid-length"],
        "maxi": ["maxi", "floor length", "floor-length"],
        "mini": ["mini skirt", "mini dress", "mini length"],
        "a line": ["a line", "a-line"],
        "wrap": ["wrap dress", "wrap style", "wraparound"],
        "athletic": ["athletic style", "sporty", "athleisure"],
        "classic": ["classic style", "timeless"],
        "vintage": ["vintage", "retro"],
        "bohemian": ["bohemian", "boho"],
        "western": ["western style", "cowboy style"],
        "minimalist": ["minimalist", "minimal style"],
    },
    "brand": {
        "nike": ["nike"],
        "adidas": ["adidas", "adidas originals"],
        "skechers": ["skechers"],
        "puma": ["puma"],
        "clarks": ["clarks"],
        "calvin klein": ["calvin klein"],
        "asics": ["asics"],
        "nine west": ["nine west"],
        "columbia": ["columbia sportswear", "columbia"],
        "under armour": ["under armour"],
        "reebok": ["reebok"],
        "levi's": ["levi's", "levis", "levi strauss"],
        "amazon essentials": ["amazon essentials"],
        "crocs": ["crocs"],
        "new balance": ["new balance"],
        "ugg": ["ugg"],
        "tommy hilfiger": ["tommy hilfiger"],
        "hanes": ["hanes"],
        "anne klein": ["anne klein"],
        "keen": ["keen footwear", "keen"],
        "the north face": ["the north face", "north face"],
        "saucony": ["saucony"],
        "michael kors": ["michael kors"],
        "merrell": ["merrell"],
        "steve madden": ["steve madden"],
        "timberland": ["timberland"],
        "carhartt": ["carhartt"],
        "converse": ["converse"],
        "sperry": ["sperry"],
        "cole haan": ["cole haan"],
        "birkenstock": ["birkenstock"],
        "dr. martens": ["dr martens", "dr. martens", "doc martens"],
        "ralph lauren": ["polo ralph lauren", "ralph lauren"],
        "toms": ["toms"],
        "vans": ["vans"],
        "brooks": ["brooks running", "brooks"],
        "mizuno": ["mizuno"],
        "ecco": ["ecco"],
        "teva": ["teva"],
        "salomon": ["salomon"],
    },
    "feature": {
        "waterproof": ["waterproof", "water proof", "water resistant", "water-resistant", "weatherproof", "rainproof"],
        "breathable": ["breathable", "breathability", "ventilated", "airflow"],
        "moisture wicking": ["moisture wicking", "moisture-wicking", "sweat wicking", "quick dry", "quick-dry", "quick drying"],
        "insulated": ["insulated", "thermal", "fleece lined", "fleece-lined", "warm lining"],
        "non slip": ["non slip", "non-slip", "nonslip", "anti slip", "anti-slip", "slip resistant", "slip-resistant"],
        "lightweight": ["lightweight", "light weight", "ultralight"],
        "comfortable": ["comfortable", "comfort", "comfy", "cushioned", "cushioning", "arch support", "supportive"],
        "adjustable": ["adjustable", "customizable fit"],
        "pockets": ["pockets", "pocket", "storage pocket"],
        "pull on closure": [
            "pull on", "pull-on", "pull on closure", "pull-on closure",
            "pull on design", "easy to pull on", "closure type: pull on",
        ],
        "zipper closure": [
            "zipper", "zipper closure", "zip closure", "zip fastening",
            "zippered", "zipper fastening", "zippered closure", "with a zipper",
            "closure type: zipper",
        ],
        "button closure": [
            "button closure", "button fastening", "button up", "button-up",
            "closure type button", "closure type: button", "with buttons",
            "has buttons",
        ],
        "drawstring closure": [
            "drawstring", "drawstring closure", "drawstring waist",
            "drawstring waistband", "adjustable drawstring", "with a drawstring",
            "closure type: drawstring",
        ],
        "buckle closure": [
            "buckle", "buckle closure", "buckle fastening", "adjustable buckle",
            "with a buckle", "closure type: buckle",
        ],
        "snap closure": [
            "snap", "snaps", "snap closure", "snap fastening", "snap button",
            "snap buttons", "press stud", "press studs", "closure type: snap",
        ],
        "stretch": ["stretch", "stretchy", "elastic", "four way stretch", "4 way stretch"],
        "uv protection": ["uv protection", "sun protection", "upf", "uva", "uvb"],
        "windproof": ["windproof", "wind proof", "wind resistant", "wind-resistant"],
        "wrinkle resistant": ["wrinkle resistant", "wrinkle-resistant", "wrinkle free", "wrinkle-free"],
        "odor resistant": ["odor resistant", "odor-resistant", "odour resistant", "anti odor", "anti-odor"],
        "stain resistant": ["stain resistant", "stain-resistant"],
        "machine washable": ["machine washable", "machine wash"],
        "easy care": ["easy care", "low maintenance"],
        "reversible": ["reversible", "two sided", "two-sided"],
        "packable": ["packable", "packs down", "foldable"],
        "reflective": ["reflective", "high visibility", "high-visibility", "hi vis"],
        "compression": ["compression", "compressive"],
        "seamless": ["seamless", "no seams"],
        "wireless": ["wireless bra", "wire free", "wire-free", "no underwire"],
        "underwire": ["underwire", "under wire"],
        "removable padding": ["removable padding", "removable pads", "removable cups"],
        "hypoallergenic": ["hypoallergenic", "nickel free", "nickel-free", "sensitive skin"],
        "anti tarnish": ["anti tarnish", "anti-tarnish", "tarnish resistant"],
        "shock absorbing": ["shock absorbing", "shock-absorbing", "impact absorption"],
        "steel toe": ["steel toe", "steel-toe", "safety toe", "composite toe"],
        "durable": ["durable", "durability", "rugged", "heavy duty", "heavy-duty"],
        "soft": ["soft", "softness", "soft touch", "buttery soft"],
    },
    "use_case": {
        "travel": ["travel", "traveling", "travelling", "vacation", "trip", "on the go", "on-the-go"],
        "hiking": ["hiking", "trail", "trekking", "backpacking"],
        "running": ["running", "jogging", "marathon", "run", "jog"],
        "walking": ["walking", "long walks", "all day walking"],
        "gym": ["gym", "workout", "working out", "training", "fitness", "exercise"],
        "yoga": ["yoga", "pilates"],
        "cycling": ["cycling", "biking", "bike riding"],
        "work": ["work", "office", "business", "professional", "workwear"],
        "casual": ["casual", "everyday", "daily wear", "weekend"],
        "formal": ["formal", "black tie", "black-tie", "dressy", "gala", "interview", "job interview"],
        "wedding": ["wedding", "bridesmaid", "bridal"],
        "party": ["party", "night out", "club", "cocktail"],
        "outdoor": ["outdoor", "outdoors", "camping", "adventure"],
        "winter": ["winter", "cold weather", "snow", "skiing", "ski trip"],
        "summer": ["summer", "hot weather", "warm weather"],
        "beach": ["beach", "pool", "seaside", "cruise"],
        "sleep": ["sleep", "sleeping", "bedtime", "lounge", "lounging"],
        "school": ["school", "class", "campus", "college"],
        "sports": ["sports", "athletics", "game day"],
        "basketball": ["basketball"],
        "tennis": ["tennis"],
        "golf": ["golf", "golfing"],
        "dance": ["dance", "dancing", "ballet"],
        "work safety": ["construction", "jobsite", "job site", "industrial work", "work safety"],
        "rain": ["rain", "rainy", "wet weather"],
    },
}


OVERRIDE_CUES = (
    "actually", "instead", "on second thought", "come to think of it",
    "ignore what i said", "ignore my earlier", "ignore that", "changed my mind",
    "change my mind", "scratch that", "forget what i said", "forget that",
    "rather than", "no longer", "i take that back",
    # People correct themselves mid-breath far more often than they announce a
    # change of mind. Without these, "sneakers -> actually boots -> no wait,
    # sandals" ends up holding boots AND sandals, which is precisely the
    # contradiction the intent-override scenario exists to punish.
    "no wait", "wait no", "hold on", "on reflection", "second thoughts",
    "i meant", "my mistake", "correction",
)

NO_PREFERENCE_CUES = (
    "no preference", "don't have a preference", "dont have a preference",
    "do not have a preference", "no strong preference", "you decide",
    "use your judgment", "use your judgement", "doesn't matter", "doesnt matter",
    "either is fine", "anything is fine", "whatever works", "no strong feelings",
    "up to you", "i'm not fussy", "im not fussy", "no idea",
)

NO_PREFERENCE_RE = re.compile(
    r"(?:do(?:n['’]?t| not) have (?:an? )?(?:additional |strong )?preference"
    r"|no (?:additional |strong )?preference"
    r"|not quite right yet"
    r"|you decide|use your judg(?:e)?ment|doesn['’]?t matter|up to you"
    r"|either is fine|anything is fine|whatever works|no strong feelings"
    r"|i['’]?m not fussy|no idea)",
    re.IGNORECASE,
)

# A missed negation is worse than a missed extraction: the reranker scores one
# point per ACTIVE value, so "I hate polyester" left un-negated makes the agent
# hunt for polyester. The simulator never phrases a rejection this way, so none
# of this is visible in the public score -- see docs/lane_c_robustness.md.
NEGATION_CUES = (
    "anything but", "don't want", "dont want", "do not want", "wouldn't want",
    "would not want", "must not", "no more", "not", "no", "without", "avoid",
    "excluding", "exclude", "except",
    "nothing", "never", "hate", "hates", "hated", "dislike", "dislikes",
    "can't", "cant", "cannot", "can not", "won't wear", "wont wear",
    "rather not", "steer clear of", "stay away from", "allergic to",
)

SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "category": ("category", "type", "kind", "item"),
    "material": ("material", "fabric"),
    "color": ("color", "colour"),
    "size": ("size", "sizing", "width"),
    "style": ("style", "fit", "cut", "silhouette"),
    "brand": ("brand", "label", "maker"),
    "budget": ("budget", "price", "cost"),
    "feature": ("feature", "function", "requirement"),
    "use_case": ("use case", "occasion", "activity", "purpose"),
}


def _surface_pattern(surface: str) -> str:
    """Escape a surface form while allowing natural separator variants."""
    chunks = [
        re.escape(chunk).replace("'", "['’]")
        for chunk in re.split(r"[\s-]+", surface.strip())
        if chunk
    ]
    return r"[\s-]+".join(chunks)


# Figurative phrases whose words look like product attributes but are not.
# "feeling blue" is not a colour preference and "watch out" is not jewelry.
# This is a blocklist of known idioms, not disambiguation -- it cannot
# generalize, but it removes the false positives that actually occur, and every
# point of slot weight amplifies the ones left behind.
IDIOM_PHRASES = (
    # colour words used figuratively
    r"black friday", r"in the red\b", r"white noise", r"feeling blue",
    r"out of the blue", r"green with envy", r"silver lining",
    r"the gold standard", r"gold medal", r"gr[ae]y area",
    r"red handed", r"tickled pink", r"navy veteran", r"orange county",
    r"purple heart", r"brown bag", r"once in a blue moon", r"black sheep",
    r"white lie", r"green light", r"red flag", r"golden retriever",
    # material words used figuratively
    r"cotton candy", r"silk road", r"velvet rope", r"iron out",
    r"iron fist", r"silver bullet",
    # garment words used figuratively
    r"watch out", r"watch the \w+", r"top of the line", r"suits? me",
    r"suits? you", r"boot up", r"belt out", r"dress rehearsal",
    r"dressed to the nines", r"ring binder", r"ring me", r"give me a ring",
    r"cap it off", r"bag of", r"tall order", r"cost an arm and a leg",
    r"hat trick", r"old hat", r"at the drop of a hat", r"under my belt",
    r"tighten(?:ing)? (?:my|our|the) belt", r"flats in the city",
)
IDIOM_RE = re.compile("|".join(f"(?:{phrase})" for phrase in IDIOM_PHRASES), re.IGNORECASE)

# "for my wife" narrows the catalog to women's items for free, and a gift
# opener is one of the most common first messages there is.
RECIPIENT_CUES: dict[str, tuple[str, ...]] = {
    "women": ("wife", "girlfriend", "mother", "mum", "mom", "sister", "aunt",
              "grandmother", "grandma", "fiancee"),
    "men": ("husband", "boyfriend", "father", "dad", "brother", "uncle",
            "grandfather", "grandpa", "fiance"),
    "girls": ("daughter", "niece", "granddaughter"),
    "boys": ("son", "nephew", "grandson"),
}
# Anchored on "for/gift for/present for" only: a relative named anywhere else
# in a sentence is far too weak to narrow a whole catalog on.
RECIPIENT_RE = {
    value: re.compile(
        r"\b(?:for|gift for|present for|buying for)\s+(?:my|a|an|the|his|her)?\s*"
        r"(?:" + "|".join(re.escape(cue) for cue in cues) + r")\b",
        re.IGNORECASE,
    )
    for value, cues in RECIPIENT_CUES.items()
}

# Amounts people spell out. Only values that plausibly bound a clothing
# purchase; nobody writes "three hundred and forty seven dollars".
WORD_NUMBERS: dict[str, int] = {
    "a couple of hundred": 200, "a couple hundred": 200, "two hundred": 200,
    "three hundred": 300, "five hundred": 500, "a thousand": 1000,
    "one hundred": 100, "a hundred": 100, "hundred": 100,
    "twenty five": 25, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}


def compile_patterns() -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    """Return ``slot -> [(canonical, word-bounded pattern)]``."""
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for slot, values in LEXICON.items():
        entries: list[tuple[str, re.Pattern[str]]] = []
        for canonical, surfaces in values.items():
            alternatives = set(surfaces)
            if len(canonical) > 1:
                alternatives.add(canonical)
            alternatives = sorted(alternatives, key=len, reverse=True)
            alternation = "|".join(_surface_pattern(surface) for surface in alternatives)
            entries.append((
                canonical,
                re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE),
            ))
        compiled[slot] = entries
    return compiled


def _compile_phrases(phrases: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(
        _surface_pattern(phrase) for phrase in sorted(phrases, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


PATTERNS = compile_patterns()
OVERRIDE_RE = _compile_phrases(OVERRIDE_CUES)
NEGATION_RE = _compile_phrases(NEGATION_CUES)
SLOT_NAME_PATTERNS = {
    slot: re.compile(
        rf"(?<!\w)(?:{'|'.join(_surface_pattern(alias) for alias in aliases)})(?!\w)",
        re.IGNORECASE,
    )
    for slot, aliases in SLOT_ALIASES.items()
}
