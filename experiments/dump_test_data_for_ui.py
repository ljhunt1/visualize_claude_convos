"""ONE-OFF TEST DATA DUMP — NOT A PIPELINE STEP.

Used to give the UI something to render while we're building it. Reads whatever
experiment artifacts happen to be on disk (labels.jsonl, conv_vecs.npz,
feature_vecs.npz, cluster labels from EXPERIMENT_DIR, transcripts from
corpus/), runs a joint UMAP, and writes a single JSON to ui/public/data.json.

When the UI grows up, the real data flow will live elsewhere; this file should
probably be deleted at that point.

Output shape is the contract with ui/src/types.ts (UIData). Conventions:
camelCase keys, "landmarks" for the reference tags overlaid on the map,
cluster=None (JSON null) for HDBSCAN noise points.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import umap

HERE = Path(__file__).parent
ROOT = HERE.parent
LABELS = HERE / "labels.jsonl"
CONV_VECS = HERE / "conv_vecs.npz"
FEATURE_VECS = HERE / "feature_vecs.npz"
CORPUS = HERE / "corpus"
EXPERIMENT_DIR = HERE / "spike_summary_tag_embed_cluster/004_tags_hdbscan_meancentered"
OUT = ROOT / "ui/public/data.json"

N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42
MEAN_CENTER = False  # joint fit: conv-only shared direction doesn't apply to landmarks


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

    lm_data = np.load(FEATURE_VECS, allow_pickle=True)
    lm_names = [str(n) for n in lm_data["names"]]
    lm_vecs = lm_data["embeddings"]
    lm_counts = lm_data["member_counts"].astype(int).tolist()
    print(f"Loaded {conv_vecs.shape[0]} convs + {lm_vecs.shape[0]} landmarks (dim {conv_vecs.shape[1]}).")

    cluster_meta = json.loads((EXPERIMENT_DIR / "meta.json").read_text())
    cluster_labels = np.load(EXPERIMENT_DIR / "labels.npy")
    cluster_by_filename = dict(zip(cluster_meta["filenames"], cluster_labels.tolist()))

    if MEAN_CENTER:
        combined = np.vstack([conv_vecs, lm_vecs])
        mean = combined.mean(axis=0, keepdims=True)
        conv_vecs = conv_vecs - mean
        lm_vecs = lm_vecs - mean
        conv_vecs = conv_vecs / np.maximum(np.linalg.norm(conv_vecs, axis=1, keepdims=True), 1e-12)
        lm_vecs = lm_vecs / np.maximum(np.linalg.norm(lm_vecs, axis=1, keepdims=True), 1e-12)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(np.vstack([conv_vecs, lm_vecs]))
    n_conv = conv_vecs.shape[0]
    conv_coords = coords[:n_conv]
    lm_coords = coords[n_conv:]

    conversations = []
    for i, fname in enumerate(conv_filenames):
        rec = records_by_filename[fname]
        date, title = parse_filename(fname)
        cluster = int(cluster_by_filename.get(fname, -1))
        transcript_path = CORPUS / fname
        conversations.append({
            "id": fname.removesuffix(".txt"),
            "title": title,
            "date": date,
            "summary": rec["summary"],
            "x": float(conv_coords[i, 0]),
            "y": float(conv_coords[i, 1]),
            "cluster": cluster if cluster >= 0 else None,
            "tags": rec["tags"],
            "nChars": rec.get("n_chars", 0),
            "transcript": transcript_path.read_text() if transcript_path.exists() else "",
        })

    landmarks = [
        {
            "name": name,
            "x": float(lm_coords[i, 0]),
            "y": float(lm_coords[i, 1]),
            "memberCount": int(lm_counts[i]),
        }
        for i, name in enumerate(lm_names)
    ]

    data = {
        "conversations": conversations,
        "landmarks": landmarks,
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "umap": {"nNeighbors": N_NEIGHBORS, "minDist": MIN_DIST, "meanCenter": MEAN_CENTER},
            "clusterSource": str(EXPERIMENT_DIR.relative_to(ROOT)),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print(f"Wrote {OUT}: {len(conversations)} convs + {len(landmarks)} landmarks.")


if __name__ == "__main__":
    main()
