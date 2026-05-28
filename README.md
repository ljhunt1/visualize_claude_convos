Need Anthropic API key in .env,

```
ANTHROPIC_API_KEY=sk-ant-...
```

**Stuff going on:**
DATA:
claude.ai -> settings -> privacy -> export data: conversation_data/ (raw imports)

LABEL SPIKE (good for messing with system prompt):

1. Extract and clean conversations: conversation_data/ ----(extract_sample.py)----> corpus/
2. Label with tags and summaries: corpus/ ----(label_spike.py)---> prints labels

END TO END SPIKE:

1. Extract and clean conversations: conversation_data/ ----(extract_sample.py)----> corpus/
2. Label with tags and summaries: corpus/ ----(label_corpus.py, imports label_spike.py)----> labels.jsonl

Pipeline: `corpus -> conv_vecs -> clusters/umaps`. Both experiment paths produce the same `conv_vecs.npz` (overwriting); the cluster step is one script that doesn't care how the vectors were built.

3a. Build conv_vecs from tag embeddings (weighted mean): labels.jsonl ----(embed_tags.py)----> conv_vecs.npz
OR
3b. Build conv_vecs from summary embeddings (direct): labels.jsonl ----(embed_summaries.py)----> conv_vecs.npz

Then for either path:

4. Cluster + report: labels.jsonl+conv_vecs.npz ----(cluster_spike.py <out_dir>)----> stdout, <out_dir>/

NEXT STEPS 5/20/26 3:10pm:

- Spiking on prompt engineering and clustering

1. Read through the experiment outcomes of 2-8 and see what I like
2. Continue to run experiments. Talk to browser claude about outcomes. Want to get good first-principles and good empirical-results way of clustering.
   Try running bert thing as a baseline and see what happens
   Try other things suggested by browser claude: Full PCA whitening; UMAP to 5d for diemnsionality reduction then HBDSCAN
   Consider bumping to 100 convos?
   Play around with system prompts to get really good summaries (see https://claude.ai/chat/a9b6c11d-ef00-4dbd-8130-c8320eea2831)
3. Mess around with UMAP scatters since this also gives me raw data on whether I think the embeddings are good

Characterizing the experiments

- Inputs: system prompt for haiku; whether I embed summaries vs tags; choice of embedding model; normalization strategy; clustering algo
- Outputs: clusters (are they high quality), UMAP (does it make sense)

---

Experiments: see [spike_summary_tag_embed_cluster/experiments.md](spike_summary_tag_embed_cluster/experiments.md).

`cluster_spike.py` takes an output directory as its sole argument and reads whatever `conv_vecs.npz` is on disk (last-written by `embed_summaries.py` or `embed_tags.py`). Each run writes `summary.md`, `conv_vecs.npy`, `labels.npy`, `meta.json` (and `mean_vec.npy` if mean-centered) into that directory. Naming convention: `NNN_<path>_<algo>_<centering>/`. Example:

```
uv run python embed_summaries.py
uv run python cluster_spike.py spike_summary_tag_embed_cluster/005_summaries_agglomerative_nomean

uv run python embed_tags.py
uv run python cluster_spike.py spike_summary_tag_embed_cluster/006_tags_agglomerative_nomean
```
