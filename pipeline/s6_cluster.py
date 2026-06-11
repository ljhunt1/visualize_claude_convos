"""Stage 6: cluster conv vectors -> clusters.json.

Mean-center + renormalize (factors out the shared "this is a Claude
conversation" direction), then HDBSCAN with cosine metric. -1 = noise.
Corpus-level: recomputed when conv_vecs change.

Usage: uv run python pipeline/cluster.py
"""
import json

import numpy as np

from config import CLUSTERS, CONV_VECS, HDBSCAN_MIN_CLUSTER_SIZE
from prompts import fingerprint


def main() -> None:
    conv = np.load(CONV_VECS, allow_pickle=True)
    input_fp = fingerprint(str(conv["input_fp"]), HDBSCAN_MIN_CLUSTER_SIZE)
    if CLUSTERS.exists():
        existing = json.loads(CLUSTERS.read_text())
        if existing.get("input_fp") == input_fp:
            print(f"clusters.json fresh (input_fp={input_fp}); skipping.")
            return

    vecs = conv["embeddings"].copy()
    mean = vecs.mean(axis=0, keepdims=True)
    vecs = vecs - mean
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

    from sklearn.cluster import HDBSCAN  # slow import; defer

    labels = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, metric="cosine").fit_predict(vecs)

    uuids = [str(u) for u in conv["uuids"]]
    by_cluster: dict[int, int] = {}
    for lbl in labels:
        by_cluster[int(lbl)] = by_cluster.get(int(lbl), 0) + 1
    n_real = len([k for k in by_cluster if k != -1])
    print(f"{n_real} clusters + {by_cluster.get(-1, 0)} noise. "
          f"Sizes: {sorted((v for k, v in by_cluster.items() if k != -1), reverse=True)}")

    CLUSTERS.write_text(json.dumps({
        "input_fp": input_fp,
        "algo": f"HDBSCAN, min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, cosine, mean-centered",
        "assignments": dict(zip(uuids, (int(l) for l in labels))),
        "names": {},  # filled in by name_clusters.py
    }, indent=2))
    print(f"Wrote {CLUSTERS.name} (names pending name_clusters.py).")


if __name__ == "__main__":
    main()
