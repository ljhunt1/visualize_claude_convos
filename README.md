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

   Embed tags:

3. Create embeddings: labels.jsonl ----(embed_tags.py)----> tag_embeddings.npz
4. Embed each vector, cluster, report: labels.jsonl+tag_embeddings.npz ----(cluster_tags.py)----> stdout, clusters.md

   Embed summaries:

3) Create embeddings: labels.jsonl ----(embed_summaries.py)---->conv_embeddings.npz
4) Embed each vector, cluster, report: labels.jsonl+conv_embeddings.npz----(cluster_summaries.py)---->stdout, cluster.md

extracting embeddings: conversation_data ---(label_corpus.py, imports label_spike.py)--->

NEXT STEPS 5/20/26 3:10pm:

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

Experiments: see [spike_outputs/experiments.md](spike_outputs/experiments.md).

`cluster_summaries.py` and `cluster_tags.py` each take an output path as their sole argument. Naming convention: `NNN_<path>_<algo>_<centering>.md`. Example:

```
uv run python cluster_summaries.py spike_outputs/005_summaries_hdbscan_nomean.md
uv run python cluster_tags.py      spike_outputs/006_tags_hdbscan_meancentered.md
```
