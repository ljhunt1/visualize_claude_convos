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

- `tag_embeddings.npz` — `tag_names: array[str]`, `embeddings: array[N, 384]`. One row per unique tag string across the corpus.
- `conv_vecs.npz` (downstream) — per-conversation aggregated vectors, computed on-demand from `labels.jsonl` + `tag_embeddings.npz`.

## Cost of dep

`sentence-transformers` pulls in `torch` (~600MB install). On a 2021 MacBook Air this is acceptable. If we ever want a lighter install, `fastembed` (Qdrant, ONNX-based) is ~50MB and serves similar models — keep in pocket.
