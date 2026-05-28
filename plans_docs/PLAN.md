# Visualize Claude Conversations — Plan

## Goal

UI showing all claude.ai conversations as a 2D scatter where semantically similar conversations cluster spatially, regions are labeled with human-readable themes, and clicking a point opens a drilldown with the conversation text + its feature scores + metadata.

Inspiration: https://transformer-circuits.pub/2024/scaling-monosemanticity/umap.html

## Per-conversation data model

- `text` (or filepath to text)
- `metadata` — datetime, project, length, turn count, code-block ratio, etc. Deterministic.
- `feature_map` — `{short_str: [0,1]}`. Open-vocabulary features (topical concepts, mood, intent, register) + their activation strengths. LLM-generated.
- `conv_vec` — dense embedding, derived from `feature_map`.

Open question: structural facts (length, code ratio) live in `metadata` for filtering. They *may* also be folded into `feature_map` as fuzzy phrase-features ("long conversation", "code-heavy") to influence layout. Worth deciding after the validation spike whether folding adds signal or just noise.

## Pipeline

1. **Feature extraction.** For each conversation, produce `feature_map`.
   - **Try first:** LLooM (Lam et al., `michelle123lam/lloom` on GitHub). Concept induction over a corpus — yields canonicalized features with per-doc scores. Built for this use case.
   - **Fallback:** Haiku 4.5 with structured-output prompting, surfacing 20–50 short tags per conversation. Prompt must explicitly request topical *and* mood/intent/register/length-feel tags, or output skews topical.
   - Adjacent prior art: TopicGPT (Pham et al.).

2. **Conversation embedding.** For each conversation:
   - Embed each feature name string with a text embedding model (Voyage `voyage-3-large`, OpenAI `text-embedding-3-large`, or local `bge-large-en`).
   - `conv_vec = (Σ_i a_i · embed(feature_i)) / (Σ_i a_i)`, L2-normalize.

3. **Layout.** UMAP the high-dim `conv_vec` → 2D coordinates.

4. **Clustering.** HDBSCAN on the **high-dim `conv_vec`**, not the UMAP output. UMAP distorts distances; cluster in the metric-preserving space.

5. **Cluster naming.** For each cluster, sample member `feature_map`s, send to Haiku, ask for a short region label.

6. **UI.**
   - Scatter plot of UMAP 2D coords, cluster region labels overlaid.
   - Click a point → drilldown: text, feature scores, metadata.
   - Sidebar: filter by metadata (datetime, project, length).
   - Later: filter/color by features (requires canonicalization — see Deferred).

## Caching & recompute rules

| Artifact | Scope | Recompute when |
|---|---|---|
| `feature_map` | per-conversation | prompt or model changes (rarely) |
| `conv_vec` | per-conversation | `feature_map` or embedding model changes |
| UMAP fit, 2D coords | corpus-level | corpus changes (new conversations loaded) |
| HDBSCAN clusters | corpus-level | corpus changes |
| Cluster labels (Haiku) | corpus-level | clusters change |

Per-conversation artifacts: compute once, essentially forever. Corpus-level artifacts: cache, invalidate on corpus growth.

For adding new conversations without a full refit, UMAP's `transform()` can project new points into the existing space — faster than refitting, slightly worse local fit. Acceptable until the corpus has grown substantially since last fit, then refit.

## Validation spike (do before scaling)

1. Run feature extraction on 10–20 conversations. Hand-iterate the prompt (or LLooM config) until tags look right.
2. Re-run on the same conversations; check feature scores are stable. If drift is bad, switch to Sonnet or tighten the rubric with anchor examples.
3. Compute embeddings for ~100 conversations, UMAP them, inspect the layout. Verify clusters make intuitive sense **before** building UI.

## Deferred

- **Feature-based filtering.** Requires a canonical feature vocabulary — a fixed set of feature IDs that raw LLM tag strings map into ("python debugging", "debugging python", "py code" → canonical `python`). LLooM produces this natively; rolled-Haiku needs an extra step (embed raw tags → HDBSCAN-cluster → LLM-name each cluster → store the mapping).
- Feature/metadata color overlays on the scatter.
- Cross-conversation feature analysis (co-occurrence, time-series of feature frequency, etc.).

## Key design decisions

- **Open vocabulary > fixed taxonomy** for features. Features discovered from data, not declared upfront. Richer, scales, more honest representation.
- **Single upstream source** for layout and labels: both derive from the LLM-generated `feature_map` → embedding chain. No independent compressions of the conversation.
- **Skip SAE activations.** Would require model access + a trained SAE. LLM-as-feature-extractor gets ~80% of the value at trivial cost.

## Reference

- **LLooM** — concept induction library. `michelle123lam/lloom` on GitHub.
- **TopicGPT** — Pham et al., adjacent technique.
- **Inspiration UI** — https://transformer-circuits.pub/2024/scaling-monosemanticity/umap.html
