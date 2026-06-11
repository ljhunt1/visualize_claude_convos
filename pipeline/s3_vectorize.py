"""Stage 3b-c: embed tags, build per-conversation vectors -> conv_vecs.npz.

conv_vec = L2-normalized score-weighted mean of the conversation's tag
embeddings (BGE-base). Tag embeddings are transient; only conv vectors
persist. Skips the (cheap but minutes-long) recompute when the labels and
embedding model are unchanged.

Usage: uv run python pipeline/vectorize.py
"""
import json

import numpy as np

from config import CONV_VECS, EMBED_MODEL
from prompts import fingerprint
from store import load_labeled_manifest, unique_tags


def is_fresh(input_fp: str) -> bool:
    if not CONV_VECS.exists():
        return False
    data = np.load(CONV_VECS, allow_pickle=True)
    return "input_fp" in data and str(data["input_fp"]) == input_fp


def conv_vec(tags: list[dict], tag_embeddings: np.ndarray, tag_idx: dict[str, int]) -> np.ndarray:
    weights = np.array([t["score"] for t in tags], dtype=np.float32)
    vecs = tag_embeddings[[tag_idx[t["name"]] for t in tags]]
    weighted = (weights[:, None] * vecs).sum(axis=0)
    norm = np.linalg.norm(weighted)
    return weighted / norm if norm > 0 else weighted


def main() -> None:
    manifest, labels = load_labeled_manifest()
    tags = unique_tags(labels)
    input_fp = fingerprint(
        EMBED_MODEL,
        json.dumps([(m["uuid"], labels[m["uuid"]]["tags"]) for m in manifest], sort_keys=True),
    )
    if is_fresh(input_fp):
        print(f"conv_vecs.npz fresh (input_fp={input_fp}); skipping.")
        return

    print(f"{len(manifest)} labeled conversations, {len(tags)} unique tags to embed.")
    from sentence_transformers import SentenceTransformer  # slow import; defer

    model = SentenceTransformer(EMBED_MODEL)
    tag_embeddings = model.encode(
        tags, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32)
    tag_idx = {name: i for i, name in enumerate(tags)}

    vecs = np.stack([conv_vec(labels[m["uuid"]]["tags"], tag_embeddings, tag_idx) for m in manifest])
    np.savez(
        CONV_VECS,
        uuids=np.array([m["uuid"] for m in manifest]),
        embeddings=vecs,
        input_fp=input_fp,
    )
    print(f"Wrote {CONV_VECS.name}: shape {vecs.shape}.")


if __name__ == "__main__":
    main()
