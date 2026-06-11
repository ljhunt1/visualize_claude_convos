"""Shared paths + model choices for the pipeline.

Stage outputs all live under pipeline/data/ (gitignored — personal data).
"""
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

EXPORT_JSON = ROOT / "conversation_data/unzipped/conversations.json"

DATA = HERE / "data"
CORPUS = DATA / "corpus"
MANIFEST = DATA / "manifest.jsonl"
LABELS = DATA / "labels.jsonl"
CONV_VECS = DATA / "conv_vecs.npz"
LANDMARKS = DATA / "landmarks.jsonl"
LANDMARKS_META = DATA / "landmarks.meta.json"
LANDMARK_VECS = DATA / "landmark_vecs.npz"
PROJECTION = DATA / "projection.npz"
CLUSTERS = DATA / "clusters.json"
UI_DATA = ROOT / "ui/public/data.json"

HAIKU_MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# Extraction filters (same as the validated experiments).
MIN_MESSAGES = 4
MAX_EMPTY_STREAK = 2

# UMAP / clustering (winning settings from spike 004).
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
HDBSCAN_MIN_CLUSTER_SIZE = 3
