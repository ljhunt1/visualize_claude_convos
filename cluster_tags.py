"""Compute per-conversation vectors, cluster with HDBSCAN, write a human-readable report.

Reads labels.jsonl + tag_embeddings.npz. Writes clusters.md and prints summary to stdout.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering

LABELS = Path("labels.jsonl")
EMBEDDINGS = Path("tag_embeddings.npz")

ALGO = "agglomerative"  # "hdbscan" or "agglomerative"
MIN_CLUSTER_SIZE = 3    # only used when ALGO == "hdbscan"
N_CLUSTERS = 8          # only used when ALGO == "agglomerative"
TOP_TAGS_PER_CLUSTER = 10
MEAN_CENTER = True  # subtract corpus centroid + renormalize; factors out the shared "this is a Claude conversation" direction


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
    if len(sys.argv) != 2:
        print(f"Usage: uv run python {sys.argv[0]} <out_path.md>")
        sys.exit(1)
    out = Path(sys.argv[1])

    records = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    data = np.load(EMBEDDINGS, allow_pickle=True)
    tag_names = list(data["tag_names"])
    embeddings = data["embeddings"]
    tag_idx = {name: i for i, name in enumerate(tag_names)}

    conv_vecs = np.stack([conv_vec(r["tags"], embeddings, tag_idx) for r in records])
    print(f"Built {conv_vecs.shape[0]} conv vectors, dim {conv_vecs.shape[1]}.")

    if MEAN_CENTER:
        mean_vec = conv_vecs.mean(axis=0, keepdims=True)
        conv_vecs = conv_vecs - mean_vec
        norms = np.linalg.norm(conv_vecs, axis=1, keepdims=True)
        conv_vecs = conv_vecs / np.maximum(norms, 1e-12)
        print(f"Mean-centered + re-normalized. Shared mean had norm {np.linalg.norm(mean_vec):.3f}.")

    if ALGO == "hdbscan":
        labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="cosine").fit_predict(conv_vecs)
        algo_desc = f"HDBSCAN, min_cluster_size={MIN_CLUSTER_SIZE}, cosine metric"
    elif ALGO == "agglomerative":
        labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="cosine", linkage="average").fit_predict(conv_vecs)
        algo_desc = f"Agglomerative, n_clusters={N_CLUSTERS}, average linkage, cosine metric"
    else:
        raise ValueError(f"Unknown ALGO: {ALGO!r}")

    clusters: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)

    n_noise = len(clusters.get(-1, []))
    n_real = len(clusters) - (1 if -1 in clusters else 0)
    sizes = sorted((len(m) for l, m in clusters.items() if l != -1), reverse=True)
    summary_line = f"{n_real} clusters" + (f" + {n_noise} noise points" if n_noise else "") + f". Sizes {sizes}."
    print(summary_line)

    lines: list[str] = [
        f"# Clusters from {len(records)} conversations",
        f"",
        f"{algo_desc} on L2-normed weighted-mean conv vectors. MEAN_CENTER={MEAN_CENTER}.",
        f"",
        f"{n_real} clusters" + (f" + {n_noise} noise points" if n_noise else "") + ".",
        f"",
    ]

    # Noise (-1) sorts last; real clusters by descending size.
    ordered = sorted(clusters.keys(), key=lambda l: (l == -1, -len(clusters[l])))
    for lbl in ordered:
        members = clusters[lbl]
        score_sum: dict[str, float] = defaultdict(float)
        count: Counter = Counter()
        for idx in members:
            for tag in records[idx]["tags"]:
                score_sum[tag["name"]] += tag["score"]
                count[tag["name"]] += 1
        top = sorted(score_sum.items(), key=lambda kv: -kv[1])[:TOP_TAGS_PER_CLUSTER]

        header = "Noise" if lbl == -1 else f"Cluster {lbl}"
        lines.append(f"## {header} — n={len(members)}")
        lines.append("")
        lines.append("**Top tags:** " + ", ".join(
            f"`{name}` ({count[name]}×, Σ={s:.1f})" for name, s in top
        ))
        lines.append("")
        lines.append("**Conversations:**")
        for idx in sorted(members, key=lambda i: records[i]["filename"]):
            lines.append(f"- {pretty_filename(records[idx]['filename'])}")
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")

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
        label_str = "noise" if lbl == -1 else str(lbl)
        print(f"{label_str:>8}  {len(members):>3}  {', '.join(top)}")


if __name__ == "__main__":
    main()
