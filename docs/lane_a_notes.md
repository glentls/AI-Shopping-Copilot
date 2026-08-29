# Lane A: understanding notes

## Measured catalog coverage

Measurements use all 50,000 rows in `data/catalog.jsonl`. The “before” column
is the thin skeleton measured immediately before the lexicon changes; the
“after” column is artifact version 3 after removing the universal
`Clothing, Shoes & Jewelry` taxonomy root.

| Slot | Before | After |
| --- | ---: | ---: |
| category | 100.0% | 99.3% |
| material | 67.9% | 78.8% |
| color | 45.1% | 48.0% |
| size | 5.4% | 22.7% |
| style | 26.8% | 49.7% |
| brand | 99.4% | 98.9% |
| budget | 0.0% | 20.8% |
| feature | 55.4% | 67.8% |
| use_case | 51.5% | 56.9% |

The brand decrease is intentional: `generic`, `unknown`, and other placeholder
stores no longer count as brands. Category coverage likewise excludes products
whose only taxonomy value is the universal root. Budget coverage reflects the
catalog's non-null numeric prices and must be used for soft ranking only.

On the same machine, a clean build took 89.0 seconds and produced a 10 MB JSON
artifact. Loading took 0.261 seconds. Across all nine slots, 25 repeated
`distribution()` calls over 5,000 candidates had medians below 0.8 ms; the
worst observed individual call was 1.58 ms.

The catalog-prose precision pass reduced `soft` from 11,411 to 11,167 products,
`work` from 7,094 to 6,259, and `classic` from 6,019 to 4,245. It also removed
47 negated `waterproof` matches and 32 negated `underwire` matches. On the
public evaluator, this changed one result from rank 3 to rank 4: Hit Rate and
MTTC were unchanged, while the technical score moved from 0.744030 to 0.743905.

Catalog confidence is based on the strongest source: store/price 1.00,
structured details 0.99, taxonomy 0.98, title 0.95, features 0.82, and
description prose 0.65. `AttributeTable.confidence()` and `.price()` expose
these optional ranking signals without changing the frozen shared contract.

## Canonical values

- **category:** women, men, girls, boys, baby, shirt, top, dress, skirt, pants,
  shorts, jeans, leggings, sweater, sweatshirt, jacket, coat, vest, suit,
  underwear, bra, sleepwear, swimwear, socks, shoes, sneakers, boots, sandals,
  heels, flats, loafers, slippers, jewelry, earrings, necklace, bracelet, ring,
  watch, bag, wallet, belt, hat, sunglasses, costume.

- **material:** cotton, polyester, leather, suede, nylon, wool, spandex, silk,
  rayon, linen, denim, fleece, acrylic, rubber, canvas, mesh, velvet, lace,
  chiffon, stainless steel, sterling silver, gold, memory foam, synthetic.

- **color:** black, white, blue, red, pink, green, brown, gray, purple, yellow,
  orange, beige, gold, silver, teal, burgundy, multicolor.

- **size:** xs, s, m, l, xl, xxl, xxxl, plus, petite, wide, narrow, tall, one
  size, plus contextual numeric shoe/clothing sizes, ranges, and bra sizes.

- **style:** long sleeve, short sleeve, three quarter sleeve, sleeveless, v
  neck, crew neck, scoop neck, turtleneck, collared, hooded, loose, regular fit,
  slim, high waisted, mid rise, low rise, pullover, zip up, button down, lace up,
  slip on, open toe, closed toe, ankle length, knee length, midi, maxi, mini, a
  line, wrap, athletic, classic, vintage, bohemian, western, minimalist.

- **brand:** nike, adidas, skechers, puma, clarks, calvin klein, asics, nine
  west, columbia, under armour, reebok, levi's, amazon essentials, crocs, new
  balance, ugg, tommy hilfiger, hanes, anne klein, keen, the north face,
  saucony, michael kors, merrell, steve madden, timberland, carhartt, converse,
  sperry, cole haan, birkenstock, dr. martens, ralph lauren, toms, vans, brooks,
  mizuno, ecco, teva, salomon. Catalog stores outside this customer-facing
  synonym list are retained as normalized dynamic brand values.

- **budget:** under 25, 25-50, 50-100, 100-200, 200+. Customer constraints stay
  numeric in `state.budget_max`; the bands exist for question distributions.

- **feature:** waterproof, breathable, moisture wicking, insulated, non slip,
  lightweight, comfortable, adjustable, pockets, stretch, uv protection,
  windproof, wrinkle resistant, odor resistant, stain resistant, machine
  washable, easy care, reversible, packable, reflective, compression,
  seamless, wireless, underwire, removable padding, hypoallergenic, anti
  tarnish, shock absorbing, steel toe, durable, soft.

- **use_case:** travel, hiking, running, walking, gym, yoga, cycling, work,
  casual, formal, wedding, party, outdoor, winter, summer, beach, sleep, school,
  sports, basketball, tennis, golf, dance, work safety, rain.

## Known failure modes

- Extraction is deterministic phrase matching, not semantic parsing. Unlisted
  paraphrases, brands, fashion subcultures, compound colors, and non-US sizing
  systems can be missed.
- Canonicalization deliberately merges nearby concepts such as navy into blue,
  cashmere into wool, and water-resistant into waterproof. This improves joins
  but loses fine-grained distinctions.
- Negation is clause-scoped. Complex constructions such as “not only black” or
  nested comparisons can still be misread.
- Catalog negation is clause-local and suppresses denied values such as “not
  waterproof” and “no underwire.” Long-distance or nested negation may still be
  missed; if the same canonical concept also has a positive mention, the
  positive evidence wins.
- Ambiguous prose-level `work` and `classic` matches require use-case/style
  context, and care instructions such as “use a soft cloth” do not tag the
  product as soft. These rules favor precision and can miss unusual wording.
- Short and polysemous terms remain risky: `small` can describe dimensions
  rather than a labeled size, `work` may appear as a verb, and names such as
  Columbia or Vans can be used non-commercially. Size matching is therefore
  restricted to titles, dedicated detail fields, and contextual numeric forms.
- Most products have no price. Missing price never means over budget, and the
  budget slot must not delete candidates.
