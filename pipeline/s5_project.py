"""Stage 5: joint UMAP of conv vectors + landmark vectors -> projection.npz.

Joint fit gives conversations and landmarks shared 2D coordinates.
Corpus-level: recomputed whenever either input changes.

Usage: uv run python pipeline/project.py
"""
import numpy as np

from config import (
    CONV_VECS,
    LANDMARK_VECS,
    PROJECTION,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)
from prompts import fingerprint


def main() -> None:
    conv = np.load(CONV_VECS, allow_pickle=True)
    lm = np.load(LANDMARK_VECS, allow_pickle=True)
    input_fp = fingerprint(
        str(conv["input_fp"]),
        lm["names"].tolist(),
        conv["embeddings"].tobytes().hex()[:64],
        lm["embeddings"].tobytes().hex()[:64],
        [UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_RANDOM_STATE],
    )
    if PROJECTION.exists():
        existing = np.load(PROJECTION, allow_pickle=True)
        if "input_fp" in existing and str(existing["input_fp"]) == input_fp:
            print(f"projection.npz fresh (input_fp={input_fp}); skipping.")
            return

    conv_vecs = conv["embeddings"]
    lm_vecs = lm["embeddings"]
    print(f"UMAP on {conv_vecs.shape[0]} convs + {lm_vecs.shape[0]} landmarks "
          f"(n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}).")

    import umap  # slow import; defer

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(np.vstack([conv_vecs, lm_vecs]))
    n_conv = conv_vecs.shape[0]

    np.savez(
        PROJECTION,
        conv_uuids=conv["uuids"],
        conv_xy=coords[:n_conv].astype(np.float32),
        landmark_names=lm["names"],
        landmark_xy=coords[n_conv:].astype(np.float32),
        input_fp=input_fp,
    )
    print(f"Wrote {PROJECTION.name}.")


if __name__ == "__main__":
    main()
