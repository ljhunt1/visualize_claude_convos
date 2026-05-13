"""Compute per-conversation vectors, cluster with HDBSCAN, write a human-readable report.

Reads labels.jsonl + tag_embeddings.npz. Writes clusters.md and prints summary to stdout.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

LABELS = Path("labels.jsonl")
EMBEDDINGS = Path("tag_embeddings.npz")
OUT = Path("clusters.md")

# Agglomerative w/ explicit k for the 50-conv spike; HDBSCAN at this size dumps
# >50% of points as noise. Revisit HDBSCAN once the corpus is in the hundreds.
N_CLUSTERS = 8
TOP_TAGS_PER_CLUSTER = 10


def conv_vec(tags: list[dict], embeddings: np.ndarray, tag_idx: dict[str, int]) -> np.ndarray:
    weights = np.array([t["score"] for t in tags], dtype=np.float32)
    vecs = embeddings[[tag_idx[t["name"]] for t in tags]]
    weighted = (weights[:, None] * vecs).sum(axis=0)
    norm = np.linalg.norm(weighted)
    return weighted / norm if norm > 0 else weighted


def pretty_filename(filename: str) -> str:
    stem = filename.removesuffix(".txt")
    # Filenames look like {uuid8}_{date}_{title}. Keep date + title.
    parts = stem.split("_", 2)
    return "  ".join(parts[1:]) if len(parts) >= 3 else stem


def main() -> None:
    records = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    data = np.load(EMBEDDINGS, allow_pickle=True)
    tag_names = list(data["tag_names"])
    embeddings = data["embeddings"]
    tag_idx = {name: i for i, name in enumerate(tag_names)}

    conv_vecs = np.stack([conv_vec(r["tags"], embeddings, tag_idx) for r in records])
    print(f"Built {conv_vecs.shape[0]} conv vectors, dim {conv_vecs.shape[1]}.")

    labels = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, metric="euclidean", linkage="average"
    ).fit_predict(conv_vecs)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)

    print(f"{len(clusters)} clusters, sizes {sorted((len(m) for m in clusters.values()), reverse=True)}.")

    lines: list[str] = [
        f"# Clusters from {len(records)} conversations",
        f"",
        f"Agglomerative clustering, n_clusters={N_CLUSTERS}, average linkage, euclidean on L2-normed weighted-mean conv vectors.",
        f"",
    ]

    ordered = sorted(clusters.keys(), key=lambda l: -len(clusters[l]))
    for lbl in ordered:
        members = clusters[lbl]
        score_sum: dict[str, float] = defaultdict(float)
        count: Counter = Counter()
        for idx in members:
            for tag in records[idx]["tags"]:
                score_sum[tag["name"]] += tag["score"]
                count[tag["name"]] += 1
        top = sorted(score_sum.items(), key=lambda kv: -kv[1])[:TOP_TAGS_PER_CLUSTER]

        lines.append(f"## Cluster {lbl} — n={len(members)}")
        lines.append("")
        lines.append("**Top tags:** " + ", ".join(
            f"`{name}` ({count[name]}×, Σ={s:.1f})" for name, s in top
        ))
        lines.append("")
        lines.append("**Conversations:**")
        for idx in sorted(members, key=lambda i: records[i]["filename"]):
            lines.append(f"- {pretty_filename(records[idx]['filename'])}")
        lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")

    # Stdout summary table
    print()
    print(f"{'cluster':>8}  {'n':>3}  top tags")
    print(f"{'-'*8}  {'-'*3}  {'-'*40}")
    for lbl in ordered:
        members = clusters[lbl]
        score_sum: dict[str, float] = defaultdict(float)
        for idx in members:
            for tag in records[idx]["tags"]:
                score_sum[tag["name"]] += tag["score"]
        top = [name for name, _ in sorted(score_sum.items(), key=lambda kv: -kv[1])[:5]]
        print(f"{lbl:>8}  {len(members):>3}  {', '.join(top)}")


if __name__ == "__main__":
    main()
