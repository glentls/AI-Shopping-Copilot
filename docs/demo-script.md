# ShopLens demo video script

Target length: 3–4 minutes. Do not show credentials, private data, or unlicensed
third-party media.

1. Introduce the problem: keyword search misses evolving intent, while a
   shopping agent must rank the purchased product quickly within ten turns.
2. State the simulator findings that shaped the design: clarification and
   recommendations share a turn; silence stalls; overrides require slot
   erasure. Narrate these — do not put an internal working document on screen.
3. Show the architecture diagram in `README.md` and explain the Buying versus
   Browsing route, recoverable constraint scoring, and offline model.
4. Run the API walkthrough. Use the interpreter that has
   `requirements-dense.lock.txt` installed — below it is `.venv-dense/bin/python`,
   matching the README setup. Either invocation works:

   ```bash
   .venv-dense/bin/python scripts/demo_session.py
   .venv-dense/bin/python -m scripts.demo_session
   ```

   The script reads the shared submission default, currently config T, instead
   of carrying a second hard-coded demo default.

   A stock `python3` without the dense dependencies is **not** usable for
   recording. Hybrid retrieval degrades to BM25 silently at the library level,
   so the demo would show BM25 results while the narration says hybrid. The
   script refuses rather than letting that happen and exits non-zero.

   If the requested/effective retrieval mismatch appears, stop and fix the
   environment; do not record around it.

   Each turn prints the customer line, the agent reply, the attribute asked,
   and the ranked Top 10 as `rank. title  parent_asin`, so a viewer can judge
   whether the ranking is good instead of reading opaque identifiers. Point out
   the accumulated requirements, the turn-three override retiring the earlier
   preference, and the zero token usage printed at the end.
5. Show the existing clean reportable T rows in `results.jsonl` and the summary
   table in `README.md`:

   ```bash
   rg -n '"config":"T"' results.jsonl
   ```

   Do not rerun holdout for the recording. The recorded T holdout is already
   exploratory, and candidate holdout access belongs only to the frozen
   evaluation gate, not the demo workflow.

   Explain HR@10, MRR, MTTC, per-scenario results, elapsed time, peak RSS,
   effective retriever, in-process vector provenance, catalog/dataset digests,
   and Git SHA.
6. Close with limitations: Boundary signal, controlled-language parsing,
   metadata sparsity, and optional LLM/cross-encoder work not claimed.

Research credit: any adapted published method named on screen must credit its
source. The surveyed ideas and their adoption boundaries are recorded in
`docs/productagent-integration.md` and `docs/wizard-of-shopping-integration.md`.
Both ablations those audits govern were measured and neither was retained, so
do not present config U or V as a shipped feature.

Before recording, replace every placeholder in `docs/devpost-draft.md`, verify
the repository is public, and confirm the final YouTube video is public.
