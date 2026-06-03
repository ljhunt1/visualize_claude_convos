"""UMAP scatter spike: reduce conv_vecs to 2D and render an interactive HTML scatter.

Reads conv_vecs.npz (whatever's on disk) + labels.jsonl. If EXPERIMENT_DIR exists,
colors points by the cluster labels saved there. Writes umap_spike.html and opens it.

Labels are persistent (truncated title above each point); hover shows full title +
summary. Plotly handles pan/zoom natively, so zoom in to read overlapping titles.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import umap

HERE = Path(__file__).parent
LABELS = HERE / "labels.jsonl"
EMBEDDINGS = HERE / "conv_vecs.npz"
EXPERIMENT_DIR = HERE / "spike_summary_tag_embed_cluster/004_tags_hdbscan_meancentered"
OUT = HERE / "umap_spike.html"

MEAN_CENTER = True
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42
LABEL_MAX_CHARS = 35


def pretty_filename(filename: str) -> str:
    stem = filename.removesuffix(".txt")
    parts = stem.split("_", 2)
    return "  ".join(parts[1:]) if len(parts) >= 3 else stem


def main() -> None:
    records_by_filename = {
        json.loads(line)["filename"]: json.loads(line)
        for line in LABELS.read_text().splitlines()
        if line.strip()
    }
    data = np.load(EMBEDDINGS, allow_pickle=True)
    filenames = [str(f) for f in data["filenames"]]
    conv_vecs = data["embeddings"]
    print(f"Loaded {conv_vecs.shape[0]} conv vectors, dim {conv_vecs.shape[1]}.")

    if MEAN_CENTER:
        mean_vec = conv_vecs.mean(axis=0, keepdims=True)
        conv_vecs = conv_vecs - mean_vec
        norms = np.linalg.norm(conv_vecs, axis=1, keepdims=True)
        conv_vecs = conv_vecs / np.maximum(norms, 1e-12)
        print(f"Mean-centered + re-normalized. Shared mean had norm {np.linalg.norm(mean_vec):.3f}.")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(conv_vecs)
    print(f"UMAP -> shape {coords.shape}.")

    cluster_strs: list[str] = ["all"] * len(filenames)
    if (EXPERIMENT_DIR / "labels.npy").exists() and (EXPERIMENT_DIR / "meta.json").exists():
        meta = json.loads((EXPERIMENT_DIR / "meta.json").read_text())
        prior_labels = np.load(EXPERIMENT_DIR / "labels.npy")
        label_by_filename = dict(zip(meta["filenames"], prior_labels.tolist()))
        cluster_strs = [
            "noise" if label_by_filename.get(f, -1) == -1 else f"cluster {label_by_filename[f]}"
            for f in filenames
        ]
        print(f"Colored by cluster labels from {EXPERIMENT_DIR.name}.")

    titles = [pretty_filename(f) for f in filenames]
    # Drop the leading date for the visible label; keep the title only, truncated.
    short_labels = [(t.split("  ", 1)[-1])[:LABEL_MAX_CHARS] for t in titles]
    summaries = [records_by_filename[f]["summary"] for f in filenames]

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "title": titles,
        "summary": summaries,
        "cluster": cluster_strs,
        "label": short_labels,
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        text="label",
        hover_name="title",
        hover_data={"summary": True, "x": False, "y": False, "label": False, "cluster": True},
        color_discrete_map={"noise": "lightgray"},
        title=(
            f"UMAP of {conv_vecs.shape[0]} conv_vecs · "
            f"MEAN_CENTER={MEAN_CENTER} · n_neighbors={N_NEIGHBORS} · min_dist={MIN_DIST}"
        ),
    )
    fig.update_traces(textposition="top center", textfont_size=9, marker=dict(size=9))
    fig.update_layout(width=1400, height=900, hovermode="closest")

    fig.write_html(OUT, auto_open=True)
    print(f"Wrote {OUT} and opened in browser.")


if __name__ == "__main__":
    main()
