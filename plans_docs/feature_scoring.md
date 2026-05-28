# Feature Scoring — Subplan

How we turn a raw conversation into a `feature_map` (`{short_str: [0,1]}`).

## Plan A — Zero-shot LLM tagging (default)

For each conversation, prompt Sonnet or Haiku to emit ~30 short descriptive tags. Use structured output (tool use w/ JSON schema) so we get a clean list back. No vocabulary management — the vocabulary is the union of every tag string the model ever emitted across the corpus.

**Why this is the bet:**
- Modern frontier models (Sonnet 4.6, Haiku 4.5) zero-shot tag-extract reliably from long contexts. LLooM-style scaffolding (distill → cluster → synthesize) was solving a problem that GPT-4o-era models had with this. Sonnet/Haiku likely don't.
- Open vocabulary by construction. No fixed taxonomy to maintain.
- Simple to bootstrap and iterate on.

**Tradeoffs accepted:**
- Vocabulary inconsistency ("frustrated debugging" / "stuck on a bug" / "user annoyed about code" — same feature, three strings). Fine for spatial layout (embeddings cluster anyway). Becomes a problem if/when we want feature-based filtering — handle with post-hoc canonicalization at that point.
- No explicit per-feature criteria. Can't audit *why* a feature fired. Acceptable for a personal exploration tool.

**Validation:** run on 5–20 conversations, eyeball the tags, iterate the prompt. If tags feel shallow, generic, or topic-only (missing mood/intent/register), tighten the prompt or switch model. Re-run on the same set to check stability.

## Plan B — LLooM (fallback)

If Plan A's tags are noisy, inconsistent, or shallow, fall back to LLooM (Lam et al., `michelle123lam/lloom`).

**What it gives us that Plan A doesn't:**
- Canonical vocabulary out of the box.
- Explicit per-concept inclusion criteria.
- Cluster-then-synthesize ensures concepts emerge from data co-occurrence, not just model priors.

**What we'd give up vs. Plan A:**
- Smaller, more constrained vocabulary (researchy, not sprawling).
- More machinery to run and tune.
- OpenAI API dependency (LLooM is GPT-4o-based by default).

**Decision rule:** stay with Plan A unless the validation spike shows clear quality problems that look like they'd be fixed by LLooM's scaffolding.
