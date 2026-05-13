Need Anthropic API key in .env,

```
ANTHROPIC_API_KEY=sk-ant-...
```

TODO:

- Tweak system prompt
- Consider

**Stuff going on:**
DATA:
claude.ai -> settings -> privacy -> export data: conversation_data/ (raw imports)

LABEL SPIKE (good for messing with system prompt):

1. Extract and clean conversations: conversation_data/ ----(extract_sample.py)----> corpus/
2. Label with tags and summaries: corpus/ ----(label_spike.py)---> prints labels

END TO END SPIKE:

1. Extract and clean conversations: conversation_data/ ----(extract_sample.py)----> corpus/
2. Label with tags and summaries: corpus/ ----(label_corpus.py, imports label_spike.py)----> labels.jsonl
3. Create embeddings: labels.jsonl ----(embed_tags.py)----> tag_embeddings.npz
4. Embed each vector, cluster, report: labels.jsonl+tag_embeddings.npz ----(cluster_spike.py)----> stdout, clusters.md

extracting embeddings: conversation_data ---(label_corpus.py, imports label_spike.py)--->
