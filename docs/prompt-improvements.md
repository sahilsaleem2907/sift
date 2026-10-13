# Precision improvements (post-benchmark)

These are the precision fixes surfaced while analysing the code-review benchmark (recall strong
~52–76%, precision collapses on dense codebases and under full agentic exploration). The
deterministic gate changes landed first as a baseline; the LLM-side prompt changes landed on top.
The prompt changes alter what the model *emits or rates* (non-deterministic), so their benchmark
effect still needs its own measurement against the deterministic baseline.

## Deterministic baseline
1. `duplicate_detect._make_finding`: `critic_exempt=False` — duplicate-logic findings now pass
   the severity gate, so `design ⇒ CONFIRMED` keeps only exact duplicates, drops fuzzy ones.
2. `severity.apply_severity_gate`: hedged / self-doubting findings (epistemic-uncertainty or
   self-negation in the body) are demoted — HIGH/CRITICAL kept as `[Unverified]`, LOW/MEDIUM
   dropped. See `_is_hedged_or_self_doubting` / `_HEDGE_PATTERNS`. This stays as the backstop for
   anything the prompt changes below miss.
3. `llm_client._annotate_diff_with_line_numbers`: per-language comment token for the `[L<n>]`
   marker (`#` / `//` / `--`), leading-gutter fallback for comment-less/markup/unknown files —
   fixes Sift flagging its own injected `// [L<n>]` as invalid syntax on YAML/JSON/etc.

## LLM-side (landed in `sift/intelligence/prompts.py`)
1. **Critic — downgrade-on-unverified.** `_KEEP_RULES`: when a claim rests on unseen/unchanged
   code and the critic checked it (tools or verification context) but still could not confirm it,
   it is kept with `certainty="speculative"` — security findings included. Folded into the
   existing "uncertainty is not a reason to drop" rule rather than added alongside it.
2. **Critic — self-consistency drop.** `_DROP_SELF_CONTRADICTION`: a finding whose own body
   concedes the problem does not occur — cleanly ("the loop does terminate") or hedged ("works,
   but is fragile", "correct only if ...") — with no concrete failing input/state is dropped. The
   `critic._apply_verdict` hard guard still keeps security/critical findings.
3. **Generator — certainty calibration.** The `REVIEW_FILE_SYSTEM` confidence rule now matches
   `schema.confidence_to_certainty` (8-10 confirmed, 7 likely, 5-6 speculative-but-reported, ≤4
   omitted — the prompt previously said "6-7 likely, 1-5 omit", so a model reporting "likely" at 6
   was silently treated as speculative and there was no band for "speculative but keep").
   Conditional/hedged findings go to 5-6 so the hedge lives in the certainty field.
4. **Generator — agentic selectivity.** Folded into the tools paragraph: verify a finding's
   triggering condition before emitting it (e.g. taint source via `find_callers`); a suspicion
   that was checked and still not confirmed is 5-6 or omitted, never "possible X if Y" at 7+.
5. **Generator — test-file discipline.** The category rule and `TEST_FILE_APPENDIX` now agree:
   test hygiene (weak/missing assertions, coverage, flakiness, naming, duplication) is always
   `maintainability`; only a defect that makes a test pass or fail wrongly (wrong expected value,
   inverted condition, never-failing assertion, wrong target) is `correctness`.
6. **Security certainty discipline.** Security is still reported aggressively, but taint-style
   findings get 8-10 only when the premise (attacker-controlled input actually reaching the sink)
   is established in the diff or by a tool; an assumed premise is 5-7. Literal credentials stay
   confirmable.

### Cleanup that came with it
- The batched and per-finding critic prompts are composed from shared rule fragments, so the two
  can no longer drift (they had been maintained as two near-identical copies).
- Removed two dangling `REVIEW_FILE_SYSTEM` bullets: one duplicated the async/IO checklist item,
  the other ("Coupling or abstraction violations") contradicted the rule directly above it that
  says pure layering observations are not findings.
- `tests/test_prompts.py` pins these invariants (shared fragments present in both critics,
  confidence bands match the code, no contradictory bullet).

## Note on bucket B (test-quality nits)
Not fixable cleanly at the gate: Sift has no `test_gap` category (that's a benchmark-side golden
label), and a path-only rule would nuke genuine golden `test_gap` matches. It is addressed by item
#5 above — a generator-categorisation fix.
