# Ablations

| Config | Hit@10 | MRR | MTTC | Efficiency | TechnicalScore | N |
|---|---|---|---|---|---|---|
| bm25_baseline | 0.1250 | 0.0680 | 9.8100 | 0.1190 | 0.1067 | 200 |
| lexical_only | 0.1700 | 0.0743 | 9.3900 | 0.1610 | 0.1395 | 200 |

| Config | Buying Hit@10 | Browsing Hit@10 | Intent Override Hit@10 | Boundary Hit@10 |
|---|---|---|---|---|
| bm25_baseline | 0.2375 | 0.0250 | 0.1333 | 0.0000 |
| lexical_only | 0.2000 | 0.1250 | 0.2000 | 0.2000 |
