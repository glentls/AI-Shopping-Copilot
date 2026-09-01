# Ablations

| Config | Hit@10 | MRR | MTTC | Efficiency | TechnicalScore | N |
|---|---|---|---|---|---|---|
| bm25_baseline | 0.1250 | 0.0680 | 9.8100 | 0.1190 | 0.1067 | 200 |
| lexical_only | 0.1700 | 0.0743 | 9.3900 | 0.1610 | 0.1395 | 200 |
| lexical_dense_rrf | 0.1400 | 0.0525 | 9.7850 | 0.1215 | 0.1100 | 200 |
| full_rrf | 0.1550 | 0.0783 | 9.6550 | 0.1345 | 0.1279 | 200 |
| full_rrf_wide | 0.1650 | 0.0820 | 9.5500 | 0.1450 | 0.1361 | 200 |

| Config | Buying Hit@10 | Browsing Hit@10 | Intent Override Hit@10 | Boundary Hit@10 |
|---|---|---|---|---|
| bm25_baseline | 0.2375 | 0.0250 | 0.1333 | 0.0000 |
| lexical_only | 0.2000 | 0.1250 | 0.2000 | 0.2000 |
| lexical_dense_rrf | 0.1375 | 0.0875 | 0.3333 | 0.0000 |
| full_rrf | 0.1625 | 0.0875 | 0.3667 | 0.0000 |
| full_rrf_wide | 0.1875 | 0.0875 | 0.3667 | 0.0000 |
