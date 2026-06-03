"""Build per-conversation vectors from tag embeddings; persist to conv_vecs.npz.

For each conversation, take its tags + scores, embed each unique tag, then compute
the L2-normed score-weighted mean of its tag vectors. Tag embeddings themselves
are transient — only the per-conversation vectors are persisted.

Output: conv_vecs.npz with arrays `filenames` (str[N]) and `embeddings` (float32[N, dim]).
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
LABELS = HERE / "labels.jsonl"
OUT = HERE / "conv_vecs.npz"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def conv_vec(tags: list[dict], tag_embeddings: np.ndarray, tag_idx: dict[str, int]) -> np.ndarray:
    weights = np.array([t["score"] for t in tags], dtype=np.float32)
    vecs = tag_embeddings[[tag_idx[t["name"]] for t in tags]]
    weighted = (weights[:, None] * vecs).sum(axis=0)
    norm = np.linalg.norm(weighted)
    return weighted / norm if norm > 0 else weighted


def main() -> None:
    records = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    filenames = [r["filename"] for r in records]

    unique_tags = sorted({t["name"] for r in records for t in r["tags"]})
    print(f"{len(records)} conversations, {len(unique_tags)} unique tags.")

    model = SentenceTransformer(MODEL_NAME)
    tag_embeddings = model.encode(
        unique_tags,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    tag_idx = {name: i for i, name in enumerate(unique_tags)}
    conv_vecs = np.stack([conv_vec(r["tags"], tag_embeddings, tag_idx) for r in records])

    np.savez(OUT, filenames=np.array(filenames), embeddings=conv_vecs)
    print(f"Wrote {OUT}: {conv_vecs.shape[0]} convos, shape {conv_vecs.shape}.")


if __name__ == "__main__":
    main()
