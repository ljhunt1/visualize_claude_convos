"""Embed every unique tag in labels.jsonl with sentence-transformers; persist to tag_embeddings.npz.

Output: tag_embeddings.npz with arrays `tag_names` (str[N]) and `embeddings` (float32[N, dim]).
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

LABELS = Path("labels.jsonl")
OUT = Path("tag_embeddings.npz")
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main() -> None:
    unique_tags: set[str] = set()
    for line in LABELS.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for tag in record["tags"]:
            unique_tags.add(tag["name"])

    tag_names = sorted(unique_tags)
    print(f"{len(tag_names)} unique tags across {LABELS}.")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        tag_names,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.savez(OUT, tag_names=np.array(tag_names), embeddings=embeddings)
    print(f"Wrote {OUT}: {embeddings.shape[0]} tags, shape {embeddings.shape}, dtype {embeddings.dtype}.")


if __name__ == "__main__":
    main()
