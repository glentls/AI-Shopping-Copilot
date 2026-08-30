# End-to-end demo runbook

Record this only after the integrated agent beats the baseline on the agreed development evaluation.

1. Run the full development evaluation and save the aggregate score and comparison with `docs/baseline_results.json`.
2. Select a deterministic buying session where the target is not already found on the first turn.
3. Capture the interaction from the initial request through the first successful Top-10 hit.
4. Make sure the recording visibly includes the structured `ask_attribute`, the shopper's answer, the narrowed recommendations, and the target `parent_asin`.
5. Label the clip with the baseline score, final score, hit turn, and note that memory is intra-session only.

The intended story is: broad request → high-value clarification → confirmed or rejected constraint is distilled → retrieval soft preferences change the ranking → target appears.
