# pipeline

The real implementation of PLAN.md's pipeline + caching rules. Stage files
are numbered (`s1_`…`s8_`) so they sort in execution order; `config.py`,
`prompts.py`, and `store.py` are shared plumbing. Every stage is
standalone-runnable and skips itself when its inputs haven't changed:

```
uv run --env-file .env python pipeline/run_all.py
```

| stage                | output (pipeline/data/)        | recomputes when                  |
| -------------------- | ------------------------------ | -------------------------------- |
| s1_extract.py        | corpus/*.txt, manifest.jsonl   | export changes                   |
| s2_label.py          | labels.jsonl (append-only)     | new convs, or prompt/model edit  |
| s3_vectorize.py      | conv_vecs.npz                  | labels or embed model change     |
| s4_landmarks.py      | landmarks.jsonl, landmark_vecs | unique-tag set changes           |
| s5_project.py        | projection.npz                 | either vector file changes       |
| s6_cluster.py        | clusters.json                  | conv_vecs change                 |
| s7_name_clusters.py  | clusters.json "names"          | cluster assignments change       |
| s8_export.py         | ../ui/public/data.json         | always (cheap)                   |

`pipeline/data/` is gitignored — it's all personal conversation data.

Prompts/tool schemas live in `prompts.py`; label caching is keyed on a
fingerprint of prompt + tool + model, so editing them relabels everything
(intentionally). `seed_from_experiments.py` is a one-off that imports the
50 spike labels as valid cache entries.
