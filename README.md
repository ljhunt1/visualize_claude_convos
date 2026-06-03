# visualize_claude_convos

UI for browsing your claude.ai conversation history as a 2D semantic scatter (UMAP-style).

See [PLAN.md](PLAN.md) for the design and roadmap.

Spike/experiment code, including earlier iterations of the embed pipeline and UI, lives in [`experiments/`](experiments/) — see [experiments/README.md](experiments/README.md). More experiments may land there over time.

## Setup

Need `ANTHROPIC_API_KEY` in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Place your unzipped claude.ai data export at `conversation_data/unzipped/conversations.json`.
