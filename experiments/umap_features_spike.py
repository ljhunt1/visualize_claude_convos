"""UMAP scatter v2 spike: features as first-class objects + conversation dots.

Joint UMAP fit over conv_vecs.npz + feature_vecs.npz so both share coordinates.
Features render as large labeled diamonds (sized by member count); conversations
render as small low-opacity dots colored by their cluster (from EXPERIMENT_DIR).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap

HERE = Path(__file__).parent
LABELS = HERE / "labels.jsonl"
CONV_EMBEDDINGS = HERE / "conv_vecs.npz"
FEATURE_EMBEDDINGS = HERE / "feature_vecs.npz"
EXPERIMENT_DIR = HERE / "spike_summary_tag_embed_cluster/004_tags_hdbscan_meancentered"
OUT = HERE / "umap_features_spike.html"

MEAN_CENTER = False  # off by default for joint fit; the conv-only "shared direction" doesn't cleanly apply to features
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42
FEATURE_LABEL_MAX = 30
CONV_LABEL_MAX = 45


def pretty_filename(filename: str) -> str:
    stem = filename.removesuffix(".txt")
    parts = stem.split("_", 2)
    return "  ".join(parts[1:]) if len(parts) >= 3 else stem


def main() -> None:
    records_by_filename = {
        json.loads(line)["filename"]: json.loads(line)
        for line in LABELS.read_text().splitlines() if line.strip()
    }

    conv_data = np.load(CONV_EMBEDDINGS, allow_pickle=True)
    conv_filenames = [str(f) for f in conv_data["filenames"]]
    conv_vecs = conv_data["embeddings"]

    feat_data = np.load(FEATURE_EMBEDDINGS, allow_pickle=True)
    feat_names = [str(n) for n in feat_data["names"]]
    feat_vecs = feat_data["embeddings"]
    feat_counts = feat_data["member_counts"].astype(int).tolist()
    print(f"Loaded {conv_vecs.shape[0]} convs + {feat_vecs.shape[0]} features (dim {conv_vecs.shape[1]}).")

    if MEAN_CENTER:
        combined = np.vstack([conv_vecs, feat_vecs])
        mean_vec = combined.mean(axis=0, keepdims=True)
        conv_vecs = conv_vecs - mean_vec
        feat_vecs = feat_vecs - mean_vec
        conv_vecs = conv_vecs / np.maximum(np.linalg.norm(conv_vecs, axis=1, keepdims=True), 1e-12)
        feat_vecs = feat_vecs / np.maximum(np.linalg.norm(feat_vecs, axis=1, keepdims=True), 1e-12)
        print(f"Mean-centered + re-normed. Combined mean had norm {np.linalg.norm(mean_vec):.3f}.")

    all_vecs = np.vstack([conv_vecs, feat_vecs])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(all_vecs)
    n_convs = conv_vecs.shape[0]
    conv_coords = coords[:n_convs]
    feat_coords = coords[n_convs:]
    print(f"UMAP -> conv {conv_coords.shape}, feat {feat_coords.shape}.")

    cluster_strs = ["unknown"] * len(conv_filenames)
    if (EXPERIMENT_DIR / "labels.npy").exists() and (EXPERIMENT_DIR / "meta.json").exists():
        meta = json.loads((EXPERIMENT_DIR / "meta.json").read_text())
        prior_labels = np.load(EXPERIMENT_DIR / "labels.npy")
        label_by_filename = dict(zip(meta["filenames"], prior_labels.tolist()))
        cluster_strs = [
            "noise" if label_by_filename.get(f, -1) == -1 else f"cluster {label_by_filename[f]}"
            for f in conv_filenames
        ]
        print(f"Conversation colors from {EXPERIMENT_DIR.name}.")

    titles = [pretty_filename(f) for f in conv_filenames]
    summaries = [records_by_filename[f]["summary"] for f in conv_filenames]
    # Drop the leading date, then truncate for the persistent label.
    conv_labels = [(t.split("  ", 1)[-1])[:CONV_LABEL_MAX] for t in titles]

    conv_df = pd.DataFrame({
        "x": conv_coords[:, 0],
        "y": conv_coords[:, 1],
        "title": titles,
        "summary": summaries,
        "cluster": cluster_strs,
        "label": conv_labels,
    })

    fig = px.scatter(
        conv_df,
        x="x", y="y",
        color="cluster",
        text="label",
        hover_name="title",
        hover_data={"summary": True, "x": False, "y": False, "cluster": True, "label": False},
        color_discrete_map={"noise": "lightgray"},
        opacity=0.75,
    )
    # Conv labels: small, gray, italic-ish, below the dot — quieter than the features.
    fig.update_traces(
        marker=dict(size=13, line=dict(width=0)),
        textposition="bottom center",
        textfont=dict(size=8, color="#888", family="Georgia, serif"),
    )

    # Features: bigger diamonds sized by member count, persistent labels on top.
    feat_sizes = [6 + np.sqrt(c) * 2.5 for c in feat_counts]
    feat_labels = [n[:FEATURE_LABEL_MAX] for n in feat_names]
    fig.add_trace(go.Scatter(
        x=feat_coords[:, 0],
        y=feat_coords[:, 1],
        mode="markers+text",
        marker=dict(symbol="diamond", size=feat_sizes, color="black", opacity=0.55,
                    line=dict(width=1, color="white")),
        text=feat_labels,
        textposition="top center",
        textfont=dict(size=11, color="black", family="Arial Black, sans-serif"),
        customdata=list(zip(feat_names, feat_counts)),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} member tag(s)<extra></extra>",
        name="features",
    ))

    fig.update_layout(
        title=(
            f"UMAP of {n_convs} conv_vecs + {feat_vecs.shape[0]} canonical features "
            f"(joint fit · MEAN_CENTER={MEAN_CENTER} · n_neighbors={N_NEIGHBORS})"
        ),
        width=1600, height=1000,
        hovermode="closest",
    )
    fig.write_html(OUT, auto_open=True)
    print(f"Wrote {OUT} and opened in browser.")


if __name__ == "__main__":
    main()
