"""Canonize tag strings via Haiku 4.5 to dedupe near-synonyms.

Reads labels.jsonl, extracts unique tags, asks Haiku in one shot to group
near-duplicates and pick a canonical name per group. Writes canonical_tags.jsonl
with one row per group: {"canonical": str, "members": list[str]}.

Requires ANTHROPIC_API_KEY in the environment.
"""
import json
from pathlib import Path

import anthropic

HERE = Path(__file__).parent
INPUT = HERE / "labels.jsonl"
OUT = HERE / "canonical_tags.jsonl"
MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Write your system prompt here. Left intentionally blank.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
"""
# ---------------------------------------------------------------------------

CANONIZE_TOOL = {
    "name": "submit_canonical_groups",
    "description": (
        "Cluster tag strings into a small number of high-level concept groups "
        "(target: 20-100 total). For each group, pick one canonical name (a "
        "member tag or a clean phrase). Every input tag must appear in exactly "
        "one group. Be aggressive — same concept under different phrasing, "
        "related sub-topics, and specific instances of a general idea all belong "
        "together."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical": {
                            "type": "string",
                            "description": "Canonical name for this group.",
                        },
                        "members": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "All input tag strings belonging to this group.",
                        },
                    },
                    "required": ["canonical", "members"],
                },
            },
        },
        "required": ["groups"],
    },
}


def main() -> None:
    unique_tags = sorted({
        t["name"]
        for line in INPUT.read_text().splitlines() if line.strip()
        for t in json.loads(line)["tags"]
    })
    print(f"{len(unique_tags)} unique tags to canonize.")

    user_msg = (
        "Cluster the following tag strings into a small number of high-level "
        "concept groups. Target: 20-100 groups total. Aggressive consolidation, "
        "NOT narrow deduping. Tags about the same broad theme — different "
        "phrasings, related sub-topics, or specific instances of a general "
        "idea — should land together.\n\n"
        "Examples of good consolidation:\n"
        '- "storage and downsizing", "local storage", "storage and logistics", '
        '"storage and belongings management" -> "storage" (or absorb into a '
        'broader "moving and logistics" that also covers relocation tags)\n'
        '- "python debugging", "pdb debugger", "python installation", '
        '"unittest framework" -> "python development"\n'
        '- "AI safety research", "AI alignment", "interpretability research", '
        '"RLHF" -> "AI safety"\n\n'
        "For each group, pick a clean canonical name (a member tag or your own "
        "phrasing). Every input tag must appear in exactly one group.\n\n"
        "Tags:\n"
        + "\n".join(f"- {t}" for t in unique_tags)
    )
    kwargs = {
        "model": MODEL,
        "max_tokens": 32000,
        "tools": [CANONIZE_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_canonical_groups"},
        "messages": [{"role": "user", "content": user_msg}],
    }
    if SYSTEM_PROMPT.strip():
        kwargs["system"] = SYSTEM_PROMPT

    client = anthropic.Anthropic()
    with client.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    print(f"stop_reason={response.stop_reason}, input_tokens={response.usage.input_tokens}, output_tokens={response.usage.output_tokens}")

    groups = None
    for block in response.content:
        if block.type == "tool_use":
            groups = block.input.get("groups")
            break
    if not groups:
        raise RuntimeError(f"no groups in tool_use; stop_reason={response.stop_reason}")

    # Save raw groups first so we never lose an expensive call.
    raw_out = OUT.with_name(OUT.stem + "_raw.json")
    raw_out.write_text(json.dumps(groups, indent=2))
    print(f"Saved raw response -> {raw_out} ({len(groups)} groups)")

    # Defensively filter malformed groups.
    good = []
    for i, g in enumerate(groups):
        if not isinstance(g.get("members"), list) or not g["members"]:
            print(f"  WARN group[{i}] malformed (canonical={g.get('canonical')!r}); dropping")
            continue
        if not isinstance(g.get("canonical"), str) or not g["canonical"]:
            print(f"  WARN group[{i}] missing canonical; dropping")
            continue
        good.append(g)

    members_seen = [t for g in good for t in g["members"]]
    missing = sorted(set(unique_tags) - set(members_seen))
    extra = sorted(set(members_seen) - set(unique_tags))
    dup_count = len(members_seen) - len(set(members_seen))
    print(f"-> {len(good)} valid groups; coverage: missing={len(missing)}, extra={len(extra)}, duplicate-assignments={dup_count}")
    if missing:
        print(f"   first missing: {missing[:8]}")
    if extra:
        print(f"   first extra: {extra[:8]}")

    # Missing tags are NOT padded into singletons — the consolidated groups are
    # the "first-class features" we want to visualize. Uncategorized tags simply
    # don't appear as their own features. (Rerun canonize with a tighter prompt
    # if you want more coverage.)
    with OUT.open("w") as f:
        for g in good:
            f.write(json.dumps(g) + "\n")
    print(f"Wrote {OUT}: {len(good)} groups ({len(missing)} tags uncategorized).")


if __name__ == "__main__":
    main()
