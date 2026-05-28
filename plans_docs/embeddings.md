# Embeddings — Notes

What we need: a way to turn each unique tag string (`"python coding"`, `"user is frustrated"`, …) into a dense vector so that semantically-similar tags land close in vector space, then a way to aggregate per conversation into `conv_vec`.

## Volume

50 conversations × ~25 tags avg ≈ 1.25k tag strings. After dedup, expect ~800–1000 unique strings. Trivial volume — fits in memory easily and embeds in seconds with any model.

## Model choice — picked: `sentence-transformers/all-MiniLM-L6-v2`

Rationale:
- Free, runs locally, no extra API key (we'd otherwise add Voyage or OpenAI).
- 384-dim vectors; plenty for short tag strings.
- ~80MB model. Fast on M1 CPU/MPS — embeds 1k strings in seconds.
- Standard baseline; quality is more than adequate for short-phrase similarity.

## Alternatives if quality bottlenecks later

| Option | When to switch | Cost |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | Slight quality bump, same footprint. Drop-in replacement. | free |
| `BAAI/bge-large-en-v1.5` | If small models can't separate fine-grained semantic neighbors. Slower. | free |
| Voyage `voyage-3-large` | Best general-purpose embedding quality. Requires a Voyage API key. | ~$0.10/M tokens |
| OpenAI `text-embedding-3-large` | Standard high-quality API option. Requires OpenAI key. | ~$0.13/M tokens |

Migration is trivial — same `tag → vector` interface, just swap the embedding call.

## Aggregation to `conv_vec`

For each conversation:

```
conv_vec = (Σ_i score_i · embedding(tag_i)) / (Σ_i score_i)
```

Then L2-normalize. This is the `feature_map` → dense-vector step from PLAN.md.

Open question: whether to transform `score_i` before the weighted mean (power scaling, temperature softmax) to sharpen the influence of high-score tags. Per the prompt-iteration discussion: store raw scores, defer the sharpening transform to layout time.

## Persistence

- `conv_vecs.npz` — `filenames: array[str]`, `embeddings: array[N, dim]`. One row per conversation. Written by either `embed_summaries.py` (direct summary embedding) or `embed_tags.py` (weighted mean of tag embeddings). Tag embeddings themselves are transient and never persisted; the conceptual pipeline is `corpus -> conv_vecs -> clusters/umaps`.

## Cost of dep

`sentence-transformers` pulls in `torch` (~600MB install). On a 2021 MacBook Air this is acceptable. If we ever want a lighter install, `fastembed` (Qdrant, ONNX-based) is ~50MB and serves similar models — keep in pocket.

## Mean-centering before clustering (important)

All conversations are human↔AI exchanges, so their embeddings share a large common direction — "this is a Claude conversation." On our 50-conv corpus, the embedding-cloud mean had norm ~0.47 (in unit-normed space), which is substantial.

Without removing this shared direction, clustering collapses most conversations into one mega-cluster: e.g. summary embeddings alone produced sizes `40, 3, 2, 1, 1, 1, 1, 1`. After **mean-centering + re-normalizing**, the same pipeline produces `12, 11, 11, 6, 4, 3, 2, 1`.

```python
mean_vec = conv_vecs.mean(axis=0, keepdims=True)
conv_vecs = conv_vecs - mean_vec
conv_vecs = conv_vecs / np.linalg.norm(conv_vecs, axis=1, keepdims=True)
```

Important detail: **pure mean-centering is a no-op for Euclidean clustering** (translation preserves pairwise distances). Re-normalizing after centering is what actually changes the angular structure — it puts vectors back on the unit sphere in the "shared-direction-removed" space, so cosine/Euclidean distances now reflect content similarity rather than meta-framing similarity.

`cluster_spike.py` applies this. Toggle via the `MEAN_CENTER` constant at the top.

## Caveat / scope

This works at N=50 with a defined corpus. If we later mix in conversations from a totally different source (e.g. ChatGPT exports), the shared direction may shift or weaken, and re-tuning will be needed.
