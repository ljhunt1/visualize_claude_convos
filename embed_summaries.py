"""Embed each conversation summary; persist to conv_embeddings.npz.

Output: conv_embeddings.npz with arrays `filenames` (str[N]) and `embeddings` (float32[N, dim]).
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

LABELS = Path("labels.jsonl")
OUT = Path("conv_embeddings.npz")
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main() -> None:
    records = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    filenames = [r["filename"] for r in records]
    summaries = [r["summary"] for r in records]
    print(f"{len(summaries)} summaries to embed.")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        summaries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.savez(OUT, filenames=np.array(filenames), embeddings=embeddings)
    print(f"Wrote {OUT}: {embeddings.shape[0]} convos, shape {embeddings.shape}.")


if __name__ == "__main__":
    main()
