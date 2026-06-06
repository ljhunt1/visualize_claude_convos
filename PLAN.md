# Visualize Claude Conversations — Plan

## Goal

UI showing all claude.ai conversations as a 2D scatter where semantically similar conversations cluster spatially, regions are labeled with human-readable themes, and clicking a point opens a drilldown with the conversation text + its feature scores + metadata.

Inspiration: https://transformer-circuits.pub/2024/scaling-monosemanticity/umap.html

## Pipeline

1. Grab raw zip of from claude.ai -> settings -> privacy -> export data. Save to `conversation_data/`
2. Extract conversations

The below steps are a work in progress, and I want to actively experiment on them:

3. Turn conversations into vectors `conversation_vecs.npz`

   a. Use haiku to generate for each conversation:
   - `feature_map` — `{short_str: [0,1]}`. Open-vocabulary features (topical concepts, mood, intent, register) + their activation strengths.
   - Few-sentence summary

   b. Embed _each tag_ using sentence transformer BAAI/bge-base-en-v1.5
   c. Each conversation's conv_vec is the weighted sum of the embedding of its tags, l2 normalized. `conv_vec = (Σ_i a_i · embed(feature_i)) / (Σ_i a_i)` -> L2-normalize.

4. Use haiku to generate a close vocabulary of "landmark" tags, which provide supplemental information in the UI. Embed each landmark as in (3). `landmark_vecs.npz`

5. UMAP the `conversation_vecs` and `landmark_vecs`
6. Cluster the `conversation_vecs`: mean-center them, then rescale to length 1 (claude tells me this is a conincal-ish transformation to apply to fix for the fact that the vectors have high anisotropy, I'm not 100% sure I get it). Then do HBDSCAN, min_cluster_size=3, cosine metric.
7. Cluster naming: TODO, should be easy just another Haiku prompt
8. UMAPS and clusters are used to build a UI
   - Scatter plot of UMAP 2D coords, cluster region labels overlaid
   - Points colored by clusters.
   - Drilldown panel per conversation (metadata, summary, transcript, tags, link to original convo)
   - Sidebar: filter by metadata (datetime, project, length)

## Places of experimentation:

- **(3); experimented on in spike_summary_tag_embed_cluster, could use more experimentation**: Generate conversation_vecs.npz by embedding summaries instead of embedding tags and taking a weighted sum. I'm not sure which of these is more "principled"
  Idea: try running `bertopic` thing as a baseline for topic tagging and see what happens
- **(3); not tried**: Use the out-of-the-box claude.ai summaries instead of our Haiku-generated summaries
- **(3); experimented on in spike_summary_tag_embed_cluster, could use more experimentation**: Choice of Haiku system prompt to generate summaries and tags is very important https://claude.ai/chat/a9b6c11d-ef00-4dbd-8130-c8320eea2831
- **(3); experimented on in spike_summary_tag_embed_cluster, could use more experimentation**: Could choose a different sentence transformer for embedding, e.g. OpenAI `text-embedding-3-large` (via API)
- **(3); not tried**: Experiment with a closed vocabulary of possible tags - maybe nice for readability or filtering down the line, and to make conversation vs landmarks less apples to oranges. Could obviate the need to generate landmarks separately, and we'd just embed landmarks + convos are weighted sums of those. Could lead to feature-based filtering
- **(6); experimented on in spike_summary_tag_embed_cluster, could use more experimentation**: Clustering algorithm. HBDSCAN vs agglomerative. Cosine metric vs not. Mean-center-then-rescale or not
  Idea: try full PCA whitening instead of mean centering
- **Not tried**: Try prompting on other things than topic or vibe, like specifically for "claude's mood and engagement style" or something. See how this correlates
- **Other analysis**: Time-series of feature frequency, co-occurence of features cross-conversation, etc.

## Data model; caching and recompute rules

| Artifact                          | Scope                      | Recompute when                                                                                        |
| --------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------- |
| conversation text                 | per-conversation, given    | N/A downloaded from claude.ai                                                                         |
| conversation metadata: datetime   | per-conversation, given    | N/A downloaded from claude.ai                                                                         |
    | N/A downloaded from claude.ai                                                                         |
| conversation metadata: word count | per-conversation, computed | When new corpus loaded from claude.ai                                                                 |
| conversation metadata: turn count | per-conversation, computed | When new corpus loaded from claude.ai                                                                 |
| conversation metadata: project    | Cant get / TODO            | N/A                                                                                                   |
| conversation metadata: model      | Cant get / TODO            | N/A                                                                                                   |
| Feature maps, summaries           | per-conversation, computed | Once on new conversations then cache; recompute when prompt or model changes (rarely)                 |
| `conv_vec`                        | per-conversation           | Once on new conversations then cache: recompute when feature maps or embedding model changes (rarely) |
| UMAP fit, 2D coords               | corpus-level               | Corpus-wide; recompute when new conversations added to corpus                                         |
| HDBSCAN clusters                  | corpus-level               | Corpus-wide; recompute when new conversations added to corpus                                         |
| Cluster labels                    | corpus-level               | Corpus-wide; recompute when new conversations added to corpus                                         |
| Landmarks and landmark embeddings | corpus-level               | Corpus-wide; recompute when new conversations added to corpus                                         |

Per-conversation artifacts: compute once, essentially forever. Corpus-level artifacts: cache, invalidate on corpus growth.

## Key design decisions

- **Open vocabulary > fixed taxonomy** for features. Features discovered from data, not declared upfront. Richer, scales, more honest representation.
- **Skip SAE activations.** Would require model access + a trained SAE. LLM-as-feature-extractor gets ~80% of the value at trivial cost.

## Reference

- **Inspiration UI** — https://transformer-circuits.pub/2024/scaling-monosemanticity/umap.html
