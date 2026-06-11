"""Stage 7: name each cluster with one Haiku call -> clusters.json "names".

Builds a digest per cluster (top tags by summed score + a few example
summaries) and asks for short distinct names. Skips when names already
exist for the current assignments.

Usage: uv run --env-file .env python pipeline/name_clusters.py
"""
import json
from collections import defaultdict

import anthropic

from config import CLUSTERS, HAIKU_MODEL
from prompts import (
    NAME_CLUSTERS_SYSTEM_PROMPT,
    NAME_CLUSTERS_TOOL,
    name_clusters_user_message,
)
from store import load_labels

TOP_TAGS = 12
EXAMPLE_SUMMARIES = 4


def main() -> None:
    clusters = json.loads(CLUSTERS.read_text())
    assignments: dict[str, int] = clusters["assignments"]
    cluster_ids = sorted({c for c in assignments.values() if c != -1})
    if clusters.get("names") and set(map(int, clusters["names"])) == set(cluster_ids):
        print("cluster names fresh; skipping.")
        return

    labels = load_labels()
    members: dict[int, list[str]] = defaultdict(list)
    for uuid, cid in assignments.items():
        if cid != -1 and uuid in labels:
            members[cid].append(uuid)

    digests = []
    for cid in cluster_ids:
        score_sum: dict[str, float] = defaultdict(float)
        for uuid in members[cid]:
            for tag in labels[uuid]["tags"]:
                score_sum[tag["name"]] += tag["score"]
        top = [name for name, _ in sorted(score_sum.items(), key=lambda kv: -kv[1])[:TOP_TAGS]]
        summaries = [labels[u]["summary"] for u in members[cid][:EXAMPLE_SUMMARIES]]
        digests.append(
            f"Cluster {cid} ({len(members[cid])} conversations)\n"
            f"Top tags: {', '.join(top)}\n"
            "Example summaries:\n" + "\n".join(f"- {s}" for s in summaries)
        )

    kwargs = {
        "model": HAIKU_MODEL,
        "max_tokens": 4096,
        "tools": [NAME_CLUSTERS_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_cluster_names"},
        "messages": [{"role": "user", "content": name_clusters_user_message(digests)}],
    }
    if NAME_CLUSTERS_SYSTEM_PROMPT.strip():
        kwargs["system"] = NAME_CLUSTERS_SYSTEM_PROMPT

    client = anthropic.Anthropic()
    response = client.messages.create(**kwargs)
    names = None
    for block in response.content:
        if block.type == "tool_use":
            names = block.input.get("names")
            break
    if not names:
        raise RuntimeError("no names in tool_use response")

    clusters["names"] = {
        str(n["cluster"]): n["name"]
        for n in names
        if isinstance(n.get("cluster"), int) and isinstance(n.get("name"), str)
    }
    missing = [c for c in cluster_ids if str(c) not in clusters["names"]]
    if missing:
        print(f"  WARN: no name returned for clusters {missing}")
    CLUSTERS.write_text(json.dumps(clusters, indent=2))
    for cid in cluster_ids:
        print(f"  cluster {cid}: {clusters['names'].get(str(cid), '???')}")
    print(f"Wrote names for {len(clusters['names'])} clusters.")


if __name__ == "__main__":
    main()
