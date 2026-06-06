"""ONE-OFF TEST DATA DUMP — NOT A PIPELINE STEP.

Used to give the UI something to render while we're building it. Reads whatever
experiment artifacts happen to be on disk (labels.jsonl, conv_vecs.npz,
feature_vecs.npz, cluster labels from EXPERIMENT_DIR), runs a joint UMAP, and
writes a single JSON to ui/public/data.json.

When the UI grows up, the real data flow will live elsewhere; this file should
probably be deleted at that point.

Output shape matches ui/src/types.ts (UIData).
"""
import json
from pathlib import Path

import numpy as np
import umap

HERE = Path(__file__).parent
ROOT = HERE.parent
LABELS = HERE / "labels.jsonl"
CONV_VECS = HERE / "conv_vecs.npz"
FEATURE_VECS = HERE / "feature_vecs.npz"
EXPERIMENT_DIR = HERE / "spike_summary_tag_embed_cluster/004_tags_hdbscan_meancentered"
OUT = ROOT / "ui/public/data.json"

N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42
MEAN_CENTER = False  # joint fit: conv-only shared direction doesn't apply to features


def parse_filename(filename: str) -> tuple[str, str]:
    """Pull (date, title) from `<uuid>_<YYYY-MM-DD>_<title_words>.txt`."""
    stem = filename.removesuffix(".txt")
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[1], parts[2].replace("_", " ")
    return "", stem


def main() -> None:
    records_by_filename = {
        json.loads(line)["filename"]: json.loads(line)
        for line in LABELS.read_text().splitlines() if line.strip()
    }

    conv_data = np.load(CONV_VECS, allow_pickle=True)
    conv_filenames = [str(f) for f in conv_data["filenames"]]
    conv_vecs = conv_data["embeddings"]

    feat_data = np.load(FEATURE_VECS, allow_pickle=True)
    feat_names = [str(n) for n in feat_data["names"]]
    feat_vecs = feat_data["embeddings"]
    feat_counts = feat_data["member_counts"].astype(int).tolist()
    print(f"Loaded {conv_vecs.shape[0]} convs + {feat_vecs.shape[0]} features (dim {conv_vecs.shape[1]}).")

    cluster_meta = json.loads((EXPERIMENT_DIR / "meta.json").read_text())
    cluster_labels = np.load(EXPERIMENT_DIR / "labels.npy")
    cluster_by_filename = dict(zip(cluster_meta["filenames"], cluster_labels.tolist()))

    if MEAN_CENTER:
        combined = np.vstack([conv_vecs, feat_vecs])
        mean = combined.mean(axis=0, keepdims=True)
        conv_vecs = conv_vecs - mean
        feat_vecs = feat_vecs - mean
        conv_vecs = conv_vecs / np.maximum(np.linalg.norm(conv_vecs, axis=1, keepdims=True), 1e-12)
        feat_vecs = feat_vecs / np.maximum(np.linalg.norm(feat_vecs, axis=1, keepdims=True), 1e-12)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(np.vstack([conv_vecs, feat_vecs]))
    n_conv = conv_vecs.shape[0]
    conv_coords = coords[:n_conv]
    feat_coords = coords[n_conv:]

    conversations = []
    for i, fname in enumerate(conv_filenames):
        rec = records_by_filename[fname]
        date, title = parse_filename(fname)
        conversations.append({
            "filename": fname,
            "title": title,
            "date": date,
            "summary": rec["summary"],
            "x": float(conv_coords[i, 0]),
            "y": float(conv_coords[i, 1]),
            "cluster": int(cluster_by_filename.get(fname, -1)),
            "tags": rec["tags"],
            "n_chars": rec.get("n_chars", 0),
        })

    features = [
        {
            "name": name,
            "x": float(feat_coords[i, 0]),
            "y": float(feat_coords[i, 1]),
            "member_count": int(feat_counts[i]),
        }
        for i, name in enumerate(feat_names)
    ]

    data = {
        "conversations": conversations,
        "features": features,
        "meta": {
            "n_conversations": len(conversations),
            "n_features": len(features),
            "umap": {"n_neighbors": N_NEIGHBORS, "min_dist": MIN_DIST, "mean_center": MEAN_CENTER},
            "cluster_source": str(EXPERIMENT_DIR.relative_to(ROOT)),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print(f"Wrote {OUT}: {len(conversations)} convs + {len(features)} features.")


if __name__ == "__main__":
    main()
