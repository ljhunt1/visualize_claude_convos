"""Stage 4: consolidate observed tags into landmark vocab; embed it.

One Haiku call groups all unique tags into 20-100 high-level concepts
(canonical name + members); each canonical name is embedded with BGE-base.
memberCount = number of *conversations* that contain at least one member
tag. Corpus-level cache: recomputed only when the unique-tag set changes.

Usage: uv run --env-file .env python pipeline/landmarks.py
"""
import json

import anthropic
import numpy as np

from config import (
    EMBED_MODEL,
    HAIKU_MODEL,
    LANDMARK_VECS,
    LANDMARKS,
    LANDMARKS_META,
)
from prompts import (
    LANDMARK_SYSTEM_PROMPT,
    LANDMARK_TOOL,
    fingerprint,
    landmark_user_message,
)
from store import load_labeled_manifest, unique_tags


def is_fresh(input_fp: str) -> bool:
    if not (LANDMARKS_META.exists() and LANDMARKS.exists() and LANDMARK_VECS.exists()):
        return False
    return json.loads(LANDMARKS_META.read_text()).get("input_fp") == input_fp


def generate_groups(tags: list[str]) -> list[dict]:
    kwargs = {
        "model": HAIKU_MODEL,
        "max_tokens": 32000,
        "tools": [LANDMARK_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_canonical_groups"},
        "messages": [{"role": "user", "content": landmark_user_message(tags)}],
    }
    if LANDMARK_SYSTEM_PROMPT.strip():
        kwargs["system"] = LANDMARK_SYSTEM_PROMPT

    client = anthropic.Anthropic()
    with client.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    print(f"stop_reason={response.stop_reason}, "
          f"in={response.usage.input_tokens}, out={response.usage.output_tokens}")

    groups = None
    for block in response.content:
        if block.type == "tool_use":
            groups = block.input.get("groups")
            break
    if not groups:
        raise RuntimeError(f"no groups in tool_use; stop_reason={response.stop_reason}")

    good = []
    for i, g in enumerate(groups):
        if not isinstance(g.get("members"), list) or not g["members"]:
            print(f"  WARN group[{i}] malformed (canonical={g.get('canonical')!r}); dropping")
            continue
        if not isinstance(g.get("canonical"), str) or not g["canonical"]:
            print(f"  WARN group[{i}] missing canonical; dropping")
            continue
        good.append({"canonical": g["canonical"], "members": g["members"]})

    members_seen = [t for g in good for t in g["members"]]
    missing = set(tags) - set(members_seen)
    print(f"-> {len(good)} valid groups; {len(missing)} tags uncategorized "
          f"(intentionally not padded into singletons).")
    return good


def main() -> None:
    manifest, labels = load_labeled_manifest()
    tags = unique_tags(labels)
    input_fp = fingerprint(
        HAIKU_MODEL, EMBED_MODEL, LANDMARK_SYSTEM_PROMPT, LANDMARK_TOOL, tags
    )
    if is_fresh(input_fp):
        print(f"landmarks fresh (input_fp={input_fp}); skipping.")
        return

    print(f"{len(tags)} unique tags from {len(manifest)} conversations to consolidate.")
    groups = generate_groups(tags)

    # memberCount = conversations containing >= 1 member tag.
    conv_counts = []
    for g in groups:
        members = set(g["members"])
        n = sum(
            1 for row in labels.values()
            if any(t["name"] in members for t in row["tags"])
        )
        conv_counts.append(n)

    names = [g["canonical"] for g in groups]
    from sentence_transformers import SentenceTransformer  # slow import; defer

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        names, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32)

    with LANDMARKS.open("w") as f:
        for g in groups:
            f.write(json.dumps(g) + "\n")
    np.savez(
        LANDMARK_VECS,
        names=np.array(names),
        embeddings=embeddings,
        member_counts=np.array(conv_counts, dtype=np.int32),
    )
    LANDMARKS_META.write_text(json.dumps({"input_fp": input_fp, "n_groups": len(groups)}))
    print(f"Wrote {LANDMARKS.name} + {LANDMARK_VECS.name}: {len(groups)} landmarks.")


if __name__ == "__main__":
    main()
