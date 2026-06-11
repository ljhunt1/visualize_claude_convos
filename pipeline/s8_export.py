"""Stage 8: join all artifacts -> ui/public/data.json.

Output shape is the contract with ui/src/types.ts (UIData): camelCase
keys, landmarks, cluster=null for noise, named clusters, claude.ai links,
full transcripts. Cheap; always runs.

Usage: uv run python pipeline/export.py
"""
import json
from datetime import datetime, timezone

import numpy as np

from config import (
    CLUSTERS,
    CORPUS,
    EMBED_MODEL,
    HAIKU_MODEL,
    PROJECTION,
    UI_DATA,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
)
from prompts import LABEL_FINGERPRINT
from store import load_labeled_manifest

LANDMARK_VECS_KEYS = ("names", "embeddings", "member_counts")


def main() -> None:
    manifest, labels = load_labeled_manifest()
    by_uuid = {m["uuid"]: m for m in manifest}

    proj = np.load(PROJECTION, allow_pickle=True)
    conv_xy = {str(u): proj["conv_xy"][i] for i, u in enumerate(proj["conv_uuids"])}

    clusters = json.loads(CLUSTERS.read_text())
    assignments: dict[str, int] = clusters["assignments"]
    names: dict[str, str] = clusters.get("names", {})

    from config import LANDMARK_VECS

    lm = np.load(LANDMARK_VECS, allow_pickle=True)
    lm_counts = lm["member_counts"].astype(int).tolist()
    lm_xy = proj["landmark_xy"]

    conversations = []
    for uuid, xy in conv_xy.items():
        entry = by_uuid.get(uuid)
        if entry is None:
            continue
        row = labels[uuid]
        cid = assignments.get(uuid, -1)
        transcript_path = CORPUS / entry["filename"]
        conversations.append({
            "id": uuid,
            "url": f"https://claude.ai/chat/{uuid}",
            "title": entry["title"],
            "date": entry["created_at"][:10],
            "summary": row["summary"],
            "x": float(xy[0]),
            "y": float(xy[1]),
            "cluster": cid if cid >= 0 else None,
            "tags": row["tags"],
            "nChars": entry["n_chars"],
            "nWords": entry["n_words"],
            "nTurns": entry["n_turns"],
            "transcript": transcript_path.read_text() if transcript_path.exists() else "",
        })
    conversations.sort(key=lambda c: c["date"])

    landmarks = [
        {
            "name": str(name),
            "x": float(lm_xy[i][0]),
            "y": float(lm_xy[i][1]),
            "memberCount": lm_counts[i],
        }
        for i, name in enumerate(proj["landmark_names"])
    ]

    cluster_sizes: dict[int, int] = {}
    for cid in assignments.values():
        if cid >= 0:
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
    cluster_list = [
        {"id": cid, "name": names.get(str(cid), f"cluster {cid}"), "size": n}
        for cid, n in sorted(cluster_sizes.items())
    ]

    data = {
        "conversations": conversations,
        "landmarks": landmarks,
        "clusters": cluster_list,
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labelModel": HAIKU_MODEL,
            "labelPromptFp": LABEL_FINGERPRINT,
            "embedModel": EMBED_MODEL,
            "umap": {"nNeighbors": UMAP_N_NEIGHBORS, "minDist": UMAP_MIN_DIST},
            "clusterAlgo": clusters["algo"],
        },
    }

    UI_DATA.parent.mkdir(parents=True, exist_ok=True)
    UI_DATA.write_text(json.dumps(data, indent=2))
    size_mb = UI_DATA.stat().st_size / 1e6
    print(f"Wrote {UI_DATA} ({size_mb:.1f} MB): {len(conversations)} convs, "
          f"{len(landmarks)} landmarks, {len(cluster_list)} clusters.")


if __name__ == "__main__":
    main()
