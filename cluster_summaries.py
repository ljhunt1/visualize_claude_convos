"""Cluster conversations by summary embedding; write a human-readable report.

Reads labels.jsonl (for tag rollups + filenames) and conv_embeddings.npz (the per-conv summary vectors).
Writes clusters_summaries.md and prints summary to stdout.

Mirrors cluster_tags.py but uses the summary-based embeddings directly rather than
aggregating per-tag vectors.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering

LABELS = Path("labels.jsonl")
EMBEDDINGS = Path("conv_embeddings.npz")

ALGO = "agglomerative"  # "hdbscan" or "agglomerative"
MIN_CLUSTER_SIZE = 3    # only used when ALGO == "hdbscan"
N_CLUSTERS = 8          # only used when ALGO == "agglomerative"
TOP_TAGS_PER_CLUSTER = 10
MEAN_CENTER = True  # subtract corpus centroid + renormalize; factors out the shared "this is a Claude conversation" direction


def pretty_filename(filename: str) -> str:
    stem = filename.removesuffix(".txt")
    parts = stem.split("_", 2)
    return "  ".join(parts[1:]) if len(parts) >= 3 else stem


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: uv run python {sys.argv[0]} <out_path.md>")
        sys.exit(1)
    out = Path(sys.argv[1])

    records_by_filename = {
        json.loads(line)["filename"]: json.loads(line)
        for line in LABELS.read_text().splitlines()
        if line.strip()
    }
    data = np.load(EMBEDDINGS, allow_pickle=True)
    filenames = [str(f) for f in data["filenames"]]
    conv_vecs = data["embeddings"]
    records = [records_by_filename[f] for f in filenames]
    print(f"Loaded {conv_vecs.shape[0]} conv vectors, dim {conv_vecs.shape[1]}.")

    if MEAN_CENTER:
        mean_vec = conv_vecs.mean(axis=0, keepdims=True)
        conv_vecs = conv_vecs - mean_vec
        # Re-normalize: pure translation is a no-op for Euclidean distance, so we put
        # vectors back on the unit sphere in the centered space (now angle-sensitive).
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
        f"# Clusters from {len(records)} conversations (summary embeddings)",
        f"",
        f"{algo_desc} on summary embeddings. MEAN_CENTER={MEAN_CENTER}.",
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
            rec = records[idx]
            lines.append(f"- **{pretty_filename(rec['filename'])}**")
            lines.append(f"  - {rec['summary']}")
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")

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
