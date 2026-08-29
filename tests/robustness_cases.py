"""Adversarial input corpus: what a person types, not what the simulator says.

The public evaluator only ever feeds the agent the simulator's stilted phrasing
("For that, what matters is: leather; 100% Leather."), so nothing in the scored
loop exercises real language. This corpus does.

Each case declares what SHOULD happen, not what currently happens, so a failure
here is a bug report rather than a snapshot. Cases marked ``soft=True`` are ones
where the right answer is genuinely debatable (non-USD currency, spelled-out
numbers); they are reported separately and never gate CI.

    python3 -m tools.robustness              # pass rate by characteristic
    python3 -m tools.robustness --failures   # every failing case
    python3 -m tools.robustness --tag numeric_not_budget
"""

from __future__ import annotations

from dataclasses import dataclass, field

SKIP = "__skip__"  # budget not asserted for this case


@dataclass(frozen=True)
class Case:
    text: str
    tag: str
    expect: tuple[tuple[str, str], ...] = ()   # (slot, value) that must be ACTIVE
    forbid: tuple[str, ...] = ()               # "slot" (no active value) or "slot:value"
    budget: object = SKIP                      # expected parse_budget result
    soft: bool = False                         # debatable; reported, never gating
    note: str = ""


def c(text, tag, expect=(), forbid=(), budget=SKIP, soft=False, note="") -> Case:
    if expect and isinstance(expect[0], str):
        expect = (expect,)
    return Case(text, tag, tuple(expect), tuple(forbid), budget, soft, note)


CASES: list[Case] = [

    # ---------------------------------------------------------------------
    # A price ceiling, stated the many ways people state one.
    # ---------------------------------------------------------------------
    c("under $80", "budget_explicit", budget=80.0),
    c("less than $50", "budget_explicit", budget=50.0),
    c("no more than $30", "budget_explicit", budget=30.0),
    c("up to $100", "budget_explicit", budget=100.0),
    c("max $45", "budget_explicit", budget=45.0),
    c("maximum of $200", "budget_explicit", budget=200.0),
    c("below $25", "budget_explicit", budget=25.0),
    c("within $60", "budget_explicit", budget=60.0),
    c("under 80 dollars", "budget_explicit", budget=80.0),
    c("less than 50 bucks", "budget_explicit", budget=50.0),
    c("my budget is $75", "budget_explicit", budget=75.0),
    c("budget of around $90", "budget_explicit", budget=90.0),
    c("around $120", "budget_explicit", budget=120.0),
    c("about $35", "budget_explicit", budget=35.0),
    c("roughly $150", "budget_explicit", budget=150.0),
    c("keep it under $15", "budget_explicit", budget=15.0),
    c("nothing over $200", "budget_explicit", budget=200.0),
    c("I don't want to spend more than $65", "budget_explicit", budget=65.0),
    c("cheaper than $40", "budget_explicit", budget=40.0),
    c("spend at most $55", "budget_explicit", budget=55.0),
    c("under $9.99", "budget_explicit", budget=9.99),
    c("under $1000", "budget_explicit", budget=1000.0),
    c("I can go up to $250 for the right one", "budget_explicit", budget=250.0),
    c("anything below $12.50 works", "budget_explicit", budget=12.5),
    c("preferably under $40, but I could stretch", "budget_explicit", budget=40.0),

    # A range names a ceiling: the HIGH end is the constraint, never the low.
    c("$20-$40", "budget_range", budget=40.0),
    c("between $30 and $60", "budget_range", budget=60.0),
    c("$10 to $25", "budget_range", budget=25.0),
    c("somewhere in the $50-$75 range", "budget_range", budget=75.0),
    c("my budget is $100 to $150", "budget_range", budget=150.0),
    c("from $15 to $30", "budget_range", budget=30.0),
    c("$45 - $90 ideally", "budget_range", budget=90.0),

    # ---------------------------------------------------------------------
    # Numbers that are NOT money. This is the reported bug class: "10 to 20
    # litres" must never become "under $10".
    # ---------------------------------------------------------------------
    c("10 to 20 litres", "numeric_not_budget", budget=None),
    c("a 30 litre backpack", "numeric_not_budget", budget=None),
    c("50 litre duffel", "numeric_not_budget", budget=None),
    c("a 24 inch necklace", "numeric_not_budget", budget=None),
    c("18 inch chain", "numeric_not_budget", budget=None),
    c("1 inch heel", "numeric_not_budget", budget=None),
    c("2 inch wide belt", "numeric_not_budget", budget=None),
    c("10 inch inseam", "numeric_not_budget", budget=None),
    c("a 6 foot scarf", "numeric_not_budget", budget=None),
    c("8mm bead", "numeric_not_budget", budget=None),
    c("size 10", "numeric_not_budget", budget=None),
    c("size 8.5", "numeric_not_budget", budget=None),
    c("size 10 to 12", "numeric_not_budget", budget=None),
    c("size 7 wide", "numeric_not_budget", budget=None),
    c("a size 4 dress", "numeric_not_budget", budget=None),
    c("size 11 shoes", "numeric_not_budget", budget=None),
    c("waist 32", "numeric_not_budget", budget=None),
    c("32x34 jeans", "numeric_not_budget", budget=None),
    c("runs 2 sizes small", "numeric_not_budget", budget=None),
    c("3 year battery", "numeric_not_budget", budget=None),
    c("2 pack of socks", "numeric_not_budget", budget=None),
    c("a 3 pack", "numeric_not_budget", budget=None),
    c("12 pairs", "numeric_not_budget", budget=None),
    c("20% off", "numeric_not_budget", budget=None),
    c("100% cotton", "numeric_not_budget", ("material", "cotton"), budget=None),
    c("60% wool 40% acrylic", "numeric_not_budget", ("material", "wool"), budget=None),
    c("5 star rating", "numeric_not_budget", budget=None),
    c("4.5 stars or better", "numeric_not_budget", budget=None),
    c("500 denier nylon", "numeric_not_budget", ("material", "nylon"), budget=None),
    c("300 thread count", "numeric_not_budget", budget=None),
    c("UV 50 protection", "numeric_not_budget", budget=None),
    c("SPF 30", "numeric_not_budget", budget=None),
    c("a 20 ounce bottle", "numeric_not_budget", budget=None),
    c("carries 15 kg", "numeric_not_budget", budget=None),
    c("a 4 person tent", "numeric_not_budget", budget=None),
    c("a 3 piece suit", "numeric_not_budget", ("category", "suit"), budget=None),
    c("14k gold", "numeric_not_budget", ("material", "gold"), budget=None),
    c("925 sterling silver", "numeric_not_budget", ("material", "sterling silver"), budget=None),
    c("aged 5 to 7", "numeric_not_budget", budget=None),
    c("model 550", "numeric_not_budget", budget=None),
    c("air max 90", "numeric_not_budget", budget=None),
    c("a 2 litre bottle holder", "numeric_not_budget", budget=None),
    c("15 to 25 degrees", "numeric_not_budget", budget=None),
    c("30 day returns", "numeric_not_budget", budget=None),
    c("ships in 2 to 3 days", "numeric_not_budget", budget=None),
    c("2 to 4 weeks of wear", "numeric_not_budget", budget=None),
    c("holds a 13 inch laptop", "numeric_not_budget", budget=None),

    # Conversational openers that happen to contain a budget trigger word.
    # "how about" carries "about"; "at most" and "max" hide in product names.
    # Found in a live session: "how about 10 to 20 litres, waterproof" was
    # being read as a $10 ceiling, which silently distorts the whole ranking.
    c("how about 10 to 20 litres, waterproof", "conversational_number",
      ("feature", "waterproof"), budget=None),
    c("how about 50", "conversational_number", budget=None),
    c("what about size 10", "conversational_number", budget=None),
    c("I'm thinking about 20 litres", "conversational_number", budget=None),
    c("tell me about the 30 litre one", "conversational_number", budget=None),
    c("how about something in cotton", "conversational_number",
      ("material", "cotton"), budget=None),
    c("what about 2 inch heels", "conversational_number", budget=None),
    c("how about a 40 litre pack", "conversational_number", budget=None),
    c("talk to me about 15 inch laptops", "conversational_number", budget=None),
    c("what do you think about 3 pairs", "conversational_number", budget=None),
    c("around the 20 litre mark", "conversational_number", budget=None),
    c("roughly 30 litres", "conversational_number", budget=None),
    c("approximately 25 cm long", "conversational_number", budget=None),
    c("air max 270", "conversational_number", budget=None),
    c("at most 3 pockets", "conversational_number", budget=None, soft=True,
      note="'at most' is budget language, but 3 pockets is a count"),

    # The same trigger words WITH a currency marker must still parse.
    c("how about something around $50", "conversational_number", budget=50.0),
    c("what about under $30", "conversational_number", budget=30.0),
    c("my budget is around 90", "conversational_number", budget=90.0),
    c("budget around fifty dollars", "conversational_number", budget=50.0),
    c("fits 5 to 10 kg", "numeric_not_budget", budget=None),

    # ---------------------------------------------------------------------
    # Negation. The value must not end up an ACTIVE preference.
    # ---------------------------------------------------------------------
    c("not leather", "negation", forbid=("material:leather",)),
    c("no polyester", "negation", forbid=("material:polyester",)),
    c("anything but black", "negation", forbid=("color:black",)),
    c("without a hood", "negation", forbid=("style:hooded",)),
    c("I don't want wool", "negation", forbid=("material:wool",)),
    c("nothing in pink", "negation", forbid=("color:pink",)),
    c("avoid synthetic materials", "negation", forbid=("material:synthetic",)),
    c("no heels please", "negation", forbid=("category:heels",)),
    c("I'd rather not have buttons", "negation", forbid=("feature:button closure",)),
    c("no zippers", "negation", forbid=("feature:zipper closure",)),
    c("definitely not white", "negation", forbid=("color:white",)),
    c("I hate the colour orange", "negation", forbid=("color:orange",)),
    c("no denim", "negation", forbid=("material:denim",)),
    c("not too slim", "negation", forbid=("style:slim",)),
    c("nothing sleeveless", "negation", forbid=("style:sleeveless",)),
    c("no wool, it itches", "negation", forbid=("material:wool",)),
    c("I can't wear nylon", "negation", forbid=("material:nylon",)),
    c("never black", "negation", forbid=("color:black",)),
    c("no lace up shoes", "negation", forbid=("style:lace up",)),
    c("anything except red", "negation", forbid=("color:red",)),
    c("not a fan of gold", "negation", forbid=("color:gold", "material:gold")),
    c("no pockets needed", "negation", forbid=("feature:pockets",)),
    c("preferably not cotton", "negation", forbid=("material:cotton",)),
    c("nothing waterproof, I don't need it", "negation", forbid=("feature:waterproof",)),

    # Negation of one thing while asserting another: only the negated one dies.
    c("not leather, cotton please", "negation_mixed",
      ("material", "cotton"), forbid=("material:leather",)),
    c("no black, I want blue", "negation_mixed",
      ("color", "blue"), forbid=("color:black",)),
    c("cotton but not white", "negation_mixed",
      ("material", "cotton"), forbid=("color:white",)),
    c("I want waterproof but not insulated", "negation_mixed",
      ("feature", "waterproof"), forbid=("feature:insulated",)),

    # ---------------------------------------------------------------------
    # Override: the new value wins, the old one is retracted.
    # ---------------------------------------------------------------------
    c("actually, make it cotton instead", "override", ("material", "cotton")),
    c("scratch that, I need boots", "override", ("category", "boots")),
    c("on second thought, blue", "override", ("color", "blue")),
    c("changed my mind, something formal", "override", ("use_case", "formal")),
    c("forget what I said, I want leather", "override", ("material", "leather")),
    c("ignore my earlier preference, waterproof is key", "override", ("feature", "waterproof")),
    c("instead of black, let's do navy", "override", ("color", "blue")),
    c("rather than wool, try fleece", "override", ("material", "fleece")),
    c("actually I changed my mind about the colour, make it green", "override",
      ("color", "green")),
    c("no longer interested in sneakers, show me boots", "override", ("category", "boots")),

    # ---------------------------------------------------------------------
    # Boundary: the customer declines to answer.
    # ---------------------------------------------------------------------
    c("you decide", "no_preference"),
    c("no preference", "no_preference"),
    c("doesn't matter", "no_preference"),
    c("up to you", "no_preference"),
    c("either is fine", "no_preference"),
    c("I'm not fussy", "no_preference"),
    c("whatever you think is best", "no_preference"),
    c("surprise me", "no_preference", soft=True),
    c("I have no strong feelings", "no_preference"),
    c("I don't mind", "no_preference"),
    c("no idea", "no_preference"),
    c("use your judgment", "no_preference"),
    c("anything works", "no_preference", soft=True),
    c("not picky", "no_preference", soft=True),

    # ---------------------------------------------------------------------
    # Idiom and figure of speech: a colour word that is not about colour, a
    # garment word that is not about a garment. The single richest source of
    # false positives in a lexicon system.
    # ---------------------------------------------------------------------
    c("black friday deal", "idiom_color", forbid=("color:black",), soft=True),
    c("in the red financially", "idiom_color", forbid=("color:red",), soft=True),
    c("white noise machine", "idiom_color", forbid=("color:white",), soft=True),
    c("feeling blue today", "idiom_color", forbid=("color:blue",), soft=True),
    c("out of the blue", "idiom_color", forbid=("color:blue",), soft=True),
    c("green with envy", "idiom_color", forbid=("color:green",), soft=True),
    c("silver lining", "idiom_color", forbid=("color:silver",), soft=True),
    c("the gold standard", "idiom_color", forbid=("color:gold",), soft=True),
    c("a grey area", "idiom_color", forbid=("color:gray",), soft=True),
    c("caught red handed", "idiom_color", forbid=("color:red",), soft=True),
    c("tickled pink", "idiom_color", forbid=("color:pink",), soft=True),
    c("a navy veteran", "idiom_color", forbid=("color:blue",), soft=True),
    c("orange county", "idiom_color", forbid=("color:orange",), soft=True),
    c("purple heart recipient", "idiom_color", forbid=("color:purple",), soft=True),
    c("brown bag lunch", "idiom_color", forbid=("color:brown",), soft=True),

    c("cotton candy costume", "idiom_material",
      ("category", "costume"), forbid=("material:cotton",), soft=True),
    c("silk road documentary", "idiom_material", forbid=("material:silk",), soft=True),
    c("iron out the details", "idiom_material", soft=True),
    c("a velvet rope event", "idiom_material", forbid=("material:velvet",), soft=True),
    c("with an iron fist", "idiom_material", soft=True),
    c("gold medal winner", "idiom_material", forbid=("material:gold",), soft=True),

    c("watch out for the price", "idiom_category", forbid=("category:watch",), soft=True),
    c("I'll watch the game later", "idiom_category", forbid=("category:watch",), soft=True),
    c("top of the line", "idiom_category", forbid=("category:top",), soft=True),
    c("that suits me fine", "idiom_category", forbid=("category:suit",), soft=True),
    c("boot up the computer", "idiom_category", forbid=("category:boots",), soft=True),
    c("belt out a song", "idiom_category", forbid=("category:belt",), soft=True),
    c("dress rehearsal", "idiom_category", forbid=("category:dress",), soft=True),
    c("a ring binder", "idiom_category", forbid=("category:ring",), soft=True),
    c("ring me later", "idiom_category", forbid=("category:ring",), soft=True),
    c("flats in the city are expensive", "idiom_category",
      forbid=("category:flats",), soft=True),
    c("cap it off with something nice", "idiom_category", forbid=("category:hat",), soft=True),
    c("a bag of chips", "idiom_category", forbid=("category:bag",), soft=True),
    c("that's a tall order", "idiom_category", forbid=("size:tall",), soft=True),
    c("wear my heart on my sleeve", "idiom_category", soft=True),
    c("dressed to the nines", "idiom_category", forbid=("category:dress",), soft=True),
    c("it cost an arm and a leg", "idiom_category", soft=True),

    # ---------------------------------------------------------------------
    # Genuinely ambiguous words used in their PRODUCT sense: must still work.
    # ---------------------------------------------------------------------
    c("tan leather boots", "ambiguous_product",
      (("color", "brown"), ("material", "leather"), ("category", "boots"))),
    c("a light jacket for spring", "ambiguous_product", ("category", "jacket")),
    c("rose gold earrings", "ambiguous_product", ("category", "earrings")),
    c("a gold ring", "ambiguous_product", ("category", "ring")),
    c("navy blue sweater", "ambiguous_product",
      (("color", "blue"), ("category", "sweater"))),
    c("charcoal grey coat", "ambiguous_product",
      (("color", "gray"), ("category", "coat"))),
    c("a wrap dress", "ambiguous_product",
      (("style", "wrap"), ("category", "dress"))),
    c("steel toe work boots", "ambiguous_product",
      (("feature", "steel toe"), ("category", "boots"))),
    c("a tank top", "ambiguous_product", ("category", "top")),
    c("cap sleeve blouse", "ambiguous_product", ("category", "top")),

    # ---------------------------------------------------------------------
    # Several constraints in one breath, which is how people actually talk.
    # ---------------------------------------------------------------------
    c("black cotton long sleeve shirt under $30 for the gym", "multi_slot",
      (("color", "black"), ("material", "cotton"), ("style", "long sleeve"),
       ("category", "shirt"), ("use_case", "gym")), budget=30.0),
    c("waterproof hiking boots in brown leather", "multi_slot",
      (("feature", "waterproof"), ("use_case", "hiking"), ("category", "boots"),
       ("color", "brown"), ("material", "leather"))),
    c("a red silk dress for a wedding under $150", "multi_slot",
      (("color", "red"), ("material", "silk"), ("category", "dress"),
       ("use_case", "wedding")), budget=150.0),
    c("lightweight breathable running shoes for summer", "multi_slot",
      (("feature", "lightweight"), ("feature", "breathable"),
       ("use_case", "running"), ("category", "shoes"), ("use_case", "summer"))),
    c("plus size high waisted jeans in dark denim", "multi_slot",
      (("size", "plus"), ("style", "high waisted"), ("category", "jeans"),
       ("material", "denim"))),
    c("a warm insulated winter coat with pockets", "multi_slot",
      (("feature", "insulated"), ("use_case", "winter"), ("category", "coat"),
       ("feature", "pockets"))),
    c("comfortable non slip work shoes for long shifts", "multi_slot",
      (("feature", "comfortable"), ("feature", "non slip"),
       ("use_case", "work"), ("category", "shoes"))),
    c("sterling silver necklace for a formal party", "multi_slot",
      (("material", "sterling silver"), ("category", "necklace"),
       ("use_case", "formal"), ("use_case", "party"))),
    c("stretchy moisture wicking yoga leggings", "multi_slot",
      (("feature", "stretch"), ("feature", "moisture wicking"),
       ("use_case", "yoga"), ("category", "leggings"))),
    c("a v neck merino wool sweater in grey", "multi_slot",
      (("style", "v neck"), ("material", "wool"), ("category", "sweater"),
       ("color", "gray"))),

    # ---------------------------------------------------------------------
    # Vague openers: the browsing case. Extracting nothing is correct;
    # hallucinating a constraint is not.
    # ---------------------------------------------------------------------
    c("just looking", "vague"),
    c("something nice", "vague"),
    c("I'm not sure what I want yet", "vague"),
    c("show me what you have", "vague"),
    c("I need something new", "vague"),
    c("browsing for now", "vague"),
    c("what's popular?", "vague"),
    c("hmm, let me think", "vague"),
    c("I'll know it when I see it", "vague"),
    c("something different", "vague"),

    # ---------------------------------------------------------------------
    # Terse: one or two words, no sentence.
    # ---------------------------------------------------------------------
    c("shoes", "terse", ("category", "shoes")),
    c("blue", "terse", ("color", "blue")),
    c("gym", "terse", ("use_case", "gym")),
    c("waterproof", "terse", ("feature", "waterproof")),
    c("cotton", "terse", ("material", "cotton")),
    c("nike", "terse", ("brand", "nike")),
    c("boots", "terse", ("category", "boots")),
    c("cheap", "terse", soft=True),
    c("large", "terse", ("size", "l")),
    c("hiking", "terse", ("use_case", "hiking")),
    c("v neck", "terse", ("style", "v neck")),
    c("black", "terse", ("color", "black")),

    # ---------------------------------------------------------------------
    # Phrased as a question, which the extractor must treat as a statement.
    # ---------------------------------------------------------------------
    c("do you have anything in blue?", "question_form", ("color", "blue")),
    c("what about leather boots?", "question_form",
      (("material", "leather"), ("category", "boots"))),
    c("can you show me cotton shirts?", "question_form",
      (("material", "cotton"), ("category", "shirt"))),
    c("is there something under $50?", "question_form", budget=50.0),
    c("got anything waterproof?", "question_form", ("feature", "waterproof")),
    c("any chance you have these in wide?", "question_form", ("size", "wide")),
    c("would you have a red dress?", "question_form",
      (("color", "red"), ("category", "dress"))),
    c("do these come in petite?", "question_form", ("size", "petite")),

    # ---------------------------------------------------------------------
    # Buried in politeness and filler.
    # ---------------------------------------------------------------------
    c("hi there, I was wondering if you could help me find some running shoes",
      "politeness", (("use_case", "running"), ("category", "shoes"))),
    c("thanks! I'm looking for a coat", "politeness", ("category", "coat")),
    c("sorry to bother you, but do you have cotton shirts?", "politeness",
      (("material", "cotton"), ("category", "shirt"))),
    c("hello, hope you're well. I need a leather belt.", "politeness",
      (("material", "leather"), ("category", "belt"))),
    c("ok so, um, I guess something in black would work", "politeness",
      ("color", "black")),
    c("great, thanks. Blue please.", "politeness", ("color", "blue")),
    c("appreciate the help! waterproof is a must", "politeness",
      ("feature", "waterproof")),

    # ---------------------------------------------------------------------
    # Long and rambling, constraint buried mid-sentence.
    # ---------------------------------------------------------------------
    c("So I've been looking for a while now and honestly nothing has really "
      "grabbed me, but what I keep coming back to is something in leather, "
      "ideally boots, that I could wear to work without them looking too casual",
      "rambling", (("material", "leather"), ("category", "boots"), ("use_case", "work"))),
    c("My old jacket finally gave up after about six winters which is honestly "
      "not bad, anyway I need a replacement, waterproof this time because I "
      "learned my lesson, and under $120 if that's possible",
      "rambling", (("category", "jacket"), ("feature", "waterproof")), budget=120.0),
    c("I'm going to Iceland in February and I have absolutely nothing warm "
      "enough, so I'm after an insulated coat, and honestly I don't care what "
      "it costs within reason",
      "rambling", (("feature", "insulated"), ("category", "coat"))),
    c("it's for my sister's wedding in June, outdoors, so probably something "
      "light and not too formal but still smart, a dress I suppose",
      "rambling", (("use_case", "wedding"), ("category", "dress"))),
    c("last pair fell apart in three months which was infuriating so this time "
      "I want something genuinely durable, leather if possible, for hiking",
      "rambling", (("feature", "durable"), ("material", "leather"), ("use_case", "hiking"))),

    # ---------------------------------------------------------------------
    # Who it is for, which implies a category segment.
    # ---------------------------------------------------------------------
    c("a gift for my wife", "recipient", ("category", "women"), soft=True),
    c("for my daughter, she's 7", "recipient", ("category", "girls"), soft=True),
    c("something for my dad", "recipient", ("category", "men"), soft=True),
    c("for a teenage boy", "recipient", ("category", "boys"), soft=True),
    c("for my grandmother", "recipient", ("category", "women"), soft=True),
    c("a present for my husband", "recipient", ("category", "men"), soft=True),
    c("baby shower gift", "recipient", ("category", "baby")),
    c("it's for me", "recipient"),

    # ---------------------------------------------------------------------
    # Season and occasion imply use_case.
    # ---------------------------------------------------------------------
    c("for winter", "occasion", ("use_case", "winter")),
    c("for a summer wedding", "occasion",
      (("use_case", "summer"), ("use_case", "wedding"))),
    c("for the beach", "occasion", ("use_case", "beach")),
    c("for my morning run", "occasion", ("use_case", "running")),
    c("something to wear to the office", "occasion", ("use_case", "work")),
    c("for a night out", "occasion", soft=True),
    c("for yoga class", "occasion", ("use_case", "yoga")),
    c("rainy season is coming", "occasion", ("use_case", "rain")),
    c("for travelling light", "occasion", ("use_case", "travel")),
    c("everyday wear", "occasion", ("use_case", "casual")),
    c("for a job interview", "occasion", ("use_case", "formal"), soft=True),
    c("going hiking next month", "occasion", ("use_case", "hiking")),

    # ---------------------------------------------------------------------
    # Comparative and superlative: relative, not absolute, constraints.
    # ---------------------------------------------------------------------
    c("the cheapest option", "comparative", soft=True),
    c("something warmer", "comparative", soft=True),
    c("more durable than the last one", "comparative", ("feature", "durable"), soft=True),
    c("the lightest one you have", "comparative", soft=True),
    c("a bit less formal", "comparative", forbid=("use_case:formal",), soft=True),
    c("slightly bigger", "comparative", soft=True),
    c("nothing too expensive", "comparative", soft=True),
    c("the most comfortable you've got", "comparative", ("feature", "comfortable")),

    # ---------------------------------------------------------------------
    # Currency other than dollars, and numbers spelled out. Debatable how far
    # a US-catalog agent should go, so all soft.
    # ---------------------------------------------------------------------
    c("under 50 euros", "currency_variant", budget=50.0, soft=True),
    c("less than 40 pounds", "currency_variant", budget=40.0, soft=True),
    c("50 quid max", "currency_variant", budget=50.0, soft=True),
    c("under £60", "currency_variant", budget=60.0, soft=True),
    c("EUR 40 or less", "currency_variant", budget=40.0, soft=True),
    c("no more than 30 USD", "currency_variant", budget=30.0, soft=True),
    c("US $30", "currency_variant", budget=30.0, soft=True),

    c("under fifty dollars", "written_number", budget=50.0, soft=True),
    c("less than a hundred bucks", "written_number", budget=100.0, soft=True),
    c("around twenty dollars", "written_number", budget=20.0, soft=True),
    c("no more than eighty", "written_number", budget=80.0, soft=True),
    c("a couple of hundred at most", "written_number", budget=200.0, soft=True),

    # ---------------------------------------------------------------------
    # Brand, including brands whose names are ordinary words.
    # ---------------------------------------------------------------------
    c("nike running shoes", "brand",
      (("brand", "nike"), ("use_case", "running"), ("category", "shoes"))),
    c("something from The North Face", "brand", ("brand", "the north face")),
    c("adidas or puma", "brand", ("brand", "adidas")),
    c("I like New Balance", "brand", ("brand", "new balance")),
    c("Dr. Martens boots", "brand", (("brand", "dr. martens"), ("category", "boots"))),
    c("levi's jeans", "brand", (("brand", "levi's"), ("category", "jeans"))),
    c("anything but Crocs", "brand", forbid=("brand:crocs",)),
    c("Vans or Converse, either is fine", "brand", soft=True),
    c("under armour compression top", "brand",
      (("brand", "under armour"), ("feature", "compression"))),
    c("I'm loyal to Merrell", "brand", ("brand", "merrell")),

    # ---------------------------------------------------------------------
    # Quantity words that must not be read as sizes or prices.
    # ---------------------------------------------------------------------
    c("a pair of boots", "quantity", ("category", "boots"), budget=None),
    c("two pairs of socks", "quantity", ("category", "socks"), budget=None),
    c("a dozen pairs", "quantity", budget=None),
    c("just one", "quantity", budget=None),
    c("a set of three", "quantity", budget=None),
    c("bulk pack", "quantity", budget=None),

    # ---------------------------------------------------------------------
    # Conditional and hedged preferences.
    # ---------------------------------------------------------------------
    c("if you have it in leather I'd prefer that", "conditional", ("material", "leather")),
    c("I'd consider wool but prefer cotton", "conditional", ("material", "cotton")),
    c("leather would be nice but isn't essential", "conditional",
      ("material", "leather"), soft=True),
    c("ideally waterproof, but I'll live without it", "conditional",
      ("feature", "waterproof"), soft=True),
    c("maybe something in blue?", "conditional", ("color", "blue")),
    c("I'm leaning towards boots", "conditional", ("category", "boots")),

    # ---------------------------------------------------------------------
    # Sentiment carrying the preference.
    # ---------------------------------------------------------------------
    c("I love blue", "sentiment", ("color", "blue")),
    c("I hate polyester", "sentiment", forbid=("material:polyester",)),
    c("leather is my favourite", "sentiment", ("material", "leather")),
    c("can't stand synthetic fabrics", "sentiment", forbid=("material:synthetic",)),
    c("I'm obsessed with anything velvet", "sentiment", ("material", "velvet")),
    c("wool always makes me itch", "sentiment", forbid=("material:wool",), soft=True),

    # ---------------------------------------------------------------------
    # Emphasis and repetition: one constraint, said twice.
    # ---------------------------------------------------------------------
    c("leather, definitely leather", "emphasis", ("material", "leather")),
    c("waterproof. Really waterproof.", "emphasis", ("feature", "waterproof")),
    c("black, and I mean black", "emphasis", ("color", "black")),
    c("comfortable comfortable comfortable", "emphasis", ("feature", "comfortable")),

    # ---------------------------------------------------------------------
    # Two intents in one message.
    # ---------------------------------------------------------------------
    c("I need boots but also maybe a belt", "mixed_intent", ("category", "boots")),
    c("a dress and matching shoes", "mixed_intent", ("category", "dress")),
    c("shirt for me, socks for my son", "mixed_intent", ("category", "shirt")),
    c("either a jacket or a coat, whichever is warmer", "mixed_intent", soft=True),

    # ---------------------------------------------------------------------
    # Size expressed the many ways people express it.
    # ---------------------------------------------------------------------
    c("extra large", "size_expression", ("size", "xl")),
    c("I'm a medium", "size_expression", ("size", "m")),
    c("plus size", "size_expression", ("size", "plus")),
    c("petite fit", "size_expression", ("size", "petite")),
    c("wide width please", "size_expression", ("size", "wide")),
    c("big and tall", "size_expression", ("size", "tall")),
    c("one size fits all", "size_expression", ("size", "one size")),
    c("XXL", "size_expression", ("size", "xxl")),
    c("small", "size_expression", ("size", "s")),
    c("narrow feet", "size_expression", ("size", "narrow")),
    c("true to size", "size_expression", soft=True),
    c("runs small so maybe size up", "size_expression", soft=True),

    # ---------------------------------------------------------------------
    # Empty and degenerate input: must never raise.
    # ---------------------------------------------------------------------
    c("", "degenerate"),
    c("   ", "degenerate"),
    c(".", "degenerate"),
    c("?", "degenerate"),
    c("ok", "degenerate"),
    c("...", "degenerate"),
    c("!!!", "degenerate"),
    c("a", "degenerate"),
    c("$", "degenerate", budget=None),
    c("$$$", "degenerate", budget=None, soft=True),
    c("100", "degenerate", budget=None, soft=True,
      note="a bare number with no currency or budget wording is not a price"),
    c("-5", "degenerate", budget=None),
    c("0", "degenerate", budget=None, soft=True),
]


# =========================================================================
# Multi-turn dialogues. A single message cannot test the three behaviours the
# score actually rewards: a preference REPLACED mid-conversation, a question
# DECLINED, and a question asked twice. These run through the real policy loop.
# =========================================================================

@dataclass(frozen=True)
class Dialogue:
    turns: tuple[str, ...]
    tag: str
    expect: tuple[tuple[str, str], ...] = ()   # active at the END of the dialogue
    forbid: tuple[str, ...] = ()               # "slot" or "slot:value", at the end
    unanswerable: tuple[str, ...] = ()         # slots that must be retired
    soft: bool = False
    note: str = ""


def d(turns, tag, expect=(), forbid=(), unanswerable=(), soft=False, note="") -> Dialogue:
    if expect and isinstance(expect[0], str):
        expect = (expect,)
    return Dialogue(tuple(turns), tag, tuple(expect), tuple(forbid),
                    tuple(unanswerable), soft, note)


DIALOGUES: list[Dialogue] = [

    # --- the override contract: the new value wins, the old one goes ------
    d(["I want a cotton shirt", "actually, make it linen instead"],
      "turn_override", ("material", "linen"), forbid=("material:cotton",)),
    d(["black boots please", "on second thought, brown"],
      "turn_override", ("color", "brown"), forbid=("color:black",)),
    d(["I need something formal", "changed my mind, casual is better"],
      "turn_override", ("use_case", "casual"), forbid=("use_case:formal",)),
    d(["leather jacket", "scratch that, denim"],
      "turn_override", ("material", "denim"), forbid=("material:leather",)),
    d(["I'd like wool", "ignore that", "cotton actually"],
      "turn_override", ("material", "cotton"), forbid=("material:wool",)),
    d(["show me sneakers", "actually boots", "no wait, sandals"],
      "turn_override", ("category", "sandals"),
      forbid=("category:sneakers", "category:boots")),

    # Re-stating a held value is emphasis, not a change. Retracting here throws
    # away a correct constraint -- the single most expensive override mistake.
    d(["I need a leather belt", "actually, what I need is leather"],
      "turn_restate", ("material", "leather"), forbid=()),
    d(["waterproof boots", "actually waterproof is the main thing"],
      "turn_restate", ("feature", "waterproof")),

    # An override must not flatten constraints it never mentioned.
    d(["black cotton shirt for work", "actually make it blue"],
      "turn_override_scope",
      (("color", "blue"), ("material", "cotton"), ("use_case", "work")),
      forbid=("color:black",),
      note="changing colour must not drop material or use_case"),
    d(["waterproof hiking boots in leather", "actually I want suede"],
      "turn_override_scope",
      (("material", "suede"), ("feature", "waterproof"), ("use_case", "hiking")),
      forbid=("material:leather",)),

    # --- retract, then revive -------------------------------------------
    d(["cotton please", "actually not cotton", "no, cotton was right"],
      "turn_revive", ("material", "cotton")),
    d(["I like blue", "not blue actually", "blue is fine after all"],
      "turn_revive", ("color", "blue"), soft=True),

    # --- negating something asserted on an earlier turn ------------------
    d(["a wool sweater", "actually no wool, it itches"],
      "turn_negate_later", forbid=("material:wool",)),
    d(["black shoes", "I don't want black after all"],
      "turn_negate_later", forbid=("color:black",)),

    # --- boundary: a declined slot is retired for good -------------------
    d(["I need boots", "I don't have a preference for material"],
      "turn_boundary", unanswerable=("material",)),
    d(["something for the gym", "no preference on colour"],
      "turn_boundary", unanswerable=("color",)),
    d(["a jacket", "you decide"], "turn_boundary"),
    d(["running shoes", "doesn't matter", "actually make them blue"],
      "turn_boundary", ("color", "blue")),

    # --- ordinary accumulation across turns ------------------------------
    d(["I need shoes", "for hiking", "waterproof if possible", "under $120"],
      "turn_accumulate",
      (("category", "shoes"), ("use_case", "hiking"), ("feature", "waterproof"))),
    d(["a dress", "for a wedding", "in red", "silk would be lovely"],
      "turn_accumulate",
      (("category", "dress"), ("use_case", "wedding"), ("color", "red"),
       ("material", "silk"))),
    d(["I'm just browsing", "maybe a coat", "something warm for winter"],
      "turn_accumulate", (("category", "coat"), ("use_case", "winter"))),
    d(["jewelry", "a necklace", "sterling silver", "nothing too long"],
      "turn_accumulate",
      (("category", "necklace"), ("material", "sterling silver"))),

    # --- budget revised across turns: the LATEST ceiling wins ------------
    d(["boots under $200", "actually I need to keep it under $80"],
      "turn_budget", note="later ceiling replaces the earlier one"),
    d(["something cheap", "up to $40 I suppose"], "turn_budget"),

    # --- the customer contradicts without any override cue ---------------
    d(["I want a black coat", "make it white"],
      "turn_implicit_conflict", ("color", "white"),
      forbid=("color:black",), soft=True,
      note="no override cue, but two colours cannot both be live"),
    d(["size small", "actually I'm a large"],
      "turn_implicit_conflict", ("size", "l"), forbid=("size:s",)),

    # --- long sessions that stay coherent --------------------------------
    d(["hi", "I need something for a trip", "shoes I think", "for walking a lot",
       "waterproof", "under $100", "not leather", "brown is fine"],
      "turn_long",
      (("category", "shoes"), ("use_case", "walking"),
       ("feature", "waterproof"), ("color", "brown")),
      forbid=("material:leather",)),
    d(["looking for a gift", "for my wife", "she likes jewelry", "a bracelet maybe",
       "silver", "under $150"],
      "turn_long", (("category", "bracelet"),)),
    d(["I need work clothes", "office not construction", "a few shirts",
       "cotton", "long sleeve", "white and blue"],
      "turn_long",
      (("category", "shirt"), ("material", "cotton"), ("style", "long sleeve"))),

    # --- declining, then volunteering anyway ------------------------------
    d(["a coat", "no preference really", "actually, waterproof matters"],
      "turn_recover", ("feature", "waterproof")),
    d(["shoes", "you decide", "oh but they must be comfortable"],
      "turn_recover", ("feature", "comfortable")),
]
