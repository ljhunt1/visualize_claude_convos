"""Embed canonical feature names; persist to feature_vecs.npz.

Reads canonical_tags.jsonl. Embeds each group's canonical name with BGE-base.

Output: feature_vecs.npz with arrays `names` (str[N]), `embeddings` (float32[N, dim]),
`member_counts` (int[N]).
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
INPUT = HERE / "canonical_tags.jsonl"
OUT = HERE / "feature_vecs.npz"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main() -> None:
    groups = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]
    names = [g["canonical"] for g in groups]
    counts = [len(g["members"]) for g in groups]
    print(f"{len(names)} canonical features to embed.")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        names,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.savez(
        OUT,
        names=np.array(names),
        embeddings=embeddings,
        member_counts=np.array(counts, dtype=np.int32),
    )
    print(f"Wrote {OUT}: {embeddings.shape[0]} features, shape {embeddings.shape}.")


if __name__ == "__main__":
    main()
