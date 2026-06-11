"""Shared loaders for pipeline artifacts."""
import json

from config import HAIKU_MODEL, LABELS, MANIFEST
from prompts import LABEL_FINGERPRINT


def load_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def load_labels() -> dict[str, dict]:
    """uuid -> label row, restricted to the current prompt + model.

    Later rows win, so a relabel of the same uuid supersedes older rows.
    """
    labels: dict[str, dict] = {}
    if not LABELS.exists():
        return labels
    for line in LABELS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["model"] == HAIKU_MODEL and row["prompt_fp"] == LABEL_FINGERPRINT:
            labels[row["uuid"]] = row
    return labels


def load_labeled_manifest() -> tuple[list[dict], dict[str, dict]]:
    """Manifest entries that have labels, plus the labels themselves."""
    labels = load_labels()
    manifest = [m for m in load_manifest() if m["uuid"] in labels]
    return manifest, labels


def unique_tags(labels: dict[str, dict]) -> list[str]:
    return sorted({t["name"] for row in labels.values() for t in row["tags"]})
