**Stuff going on:**
DATA:
claude.ai -> settings -> privacy -> export data: conversation_data/ (raw imports)

### Label Spike (good for messing with system prompt):

1. Extract and clean conversations: conversation_data/ ----(extract_sample.py)----> corpus/
2. Label with tags and summaries: corpus/ ----(label_spike.py)---> prints labels

### Summary/tag/embed spike (test systems prompts + clustering algos end to end):

`conversation_data -> corpus ---(tags or summaries)---> conv_vecs -> clusters/umaps`.

1. Extract and clean conversations: conversation_data/ ----(extract_sample.py)----> corpus/
2. Label with tags and summaries: corpus/ ----(label_corpus.py, imports label_spike.py)----> labels.jsonl'

3a. Build conv_vecs from tag embeddings (weighted mean): labels.jsonl ----(embed_tags.py)----> conv_vecs.npz
OR
3b. Build conv_vecs from summary embeddings (direct): labels.jsonl ----(embed_summaries.py)----> conv_vecs.npz

4. Cluster + report: labels.jsonl+conv_vecs.npz ----(cluster_spike.py <out_dir>)----> stdout, <out_dir>/

### UI spike (test system prompt + clustering algos -> )

1. labels.jsonl ----(canonize_tags.py)---> canonical_tags.jsonl # Haiku makes canonical "landmark" tags
2. canonical_tags.jsonl ----(embed_features.py)---->feature_vecs.npz # Embed the features
3. UI: umap_spike.py -> umap_spike.html (takes conv_vecs and feature_vecs, embeds and UMAPs)
   OR umap_features_spike.py -> umap_features_spike.html

Basically experiments are:
Inputs: system prompt for haiku; whether I embed summaries vs tags; choice of embedding model; normalization strategy; clustering algo
Outputs: clusters (are they high quality), UMAP (does it make sense)

And also messing with the UI

---

Experiments: see [spike_summary_tag_embed_cluster/experiments.md](spike_summary_tag_embed_cluster/experiments.md).

`cluster_spike.py` takes an output directory as its sole argument and reads whatever `conv_vecs.npz` is on disk (last-written by `embed_summaries.py` or `embed_tags.py`). Each run writes `summary.md`, `conv_vecs.npy`, `labels.npy`, `meta.json` (and `mean_vec.npy` if mean-centered) into that directory. Naming convention: `NNN_<path>_<algo>_<centering>/`. Example:

Run from repo root (scripts resolve their data paths relative to `experiments/`):

```
uv run --env-file=.env python experiments/embed_summaries.py
uv run --env-file=.env python experiments/cluster_spike.py experiments/spike_summary_tag_embed_cluster/005_summaries_agglomerative_nomean

uv run --env-file=.env python experiments/embed_tags.py
uv run --env-file=.env python experiments/cluster_spike.py experiments/spike_summary_tag_embed_cluster/006_tags_agglomerative_nomean
```
