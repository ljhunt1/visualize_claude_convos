"""Extract a time-spread sample of conversations from the unzipped Claude export.

Usage: uv run python extract_sample.py [N] [OUT_DIR]
  N        number of conversations to sample (default 5)
  OUT_DIR  output directory (default corpus)
"""
import json
import sys
from pathlib import Path

EXPORT = Path("conversation_data/unzipped/conversations.json")
MIN_MESSAGES = 4
MAX_EMPTY_STREAK = 2  # reject convos with 3+ consecutive empty messages


def conversation_to_text(conv: dict) -> str:
    title = conv["name"] or "(untitled)"
    lines = [f"# {title}", f"Created: {conv['created_at']}", ""]
    for msg in conv["chat_messages"]:
        lines.append(f"## {msg['sender'].upper()}")
        lines.append((msg.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines)


def safe_filename(s: str, max_len: int = 60) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in s).strip()
    return (cleaned or "untitled")[:max_len].replace(" ", "_")


def has_empty_streak(conv: dict, limit: int) -> bool:
    """True if conv has more than `limit` consecutive empty-text messages."""
    streak = 0
    for m in conv["chat_messages"]:
        if not (m.get("text") or "").strip():
            streak += 1
            if streak > limit:
                return True
        else:
            streak = 0
    return False


def extract(n_samples: int, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    convs = json.loads(EXPORT.read_text())
    eligible = [
        c for c in convs
        if len(c["chat_messages"]) >= MIN_MESSAGES
        and not has_empty_streak(c, MAX_EMPTY_STREAK)
    ]
    eligible.sort(key=lambda c: c["created_at"])

    n = len(eligible)
    print(f"{len(convs)} total; {n} pass filters (>= {MIN_MESSAGES} msgs, no streak of >{MAX_EMPTY_STREAK} empty).")

    if n_samples >= n:
        sampled = eligible
    else:
        indices = [round(i * (n - 1) / (n_samples - 1)) for i in range(n_samples)]
        sampled = [eligible[i] for i in indices]

    for c in sampled:
        name = safe_filename(c["name"] or c["uuid"][:8])
        # Prefix with UUID short-hash so labels.jsonl can join cleanly later.
        path = out_dir / f"{c['uuid'][:8]}_{c['created_at'][:10]}_{name}.txt"
        path.write_text(conversation_to_text(c))
    print(f"Wrote {len(sampled)} files to {out_dir}/")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus")
    extract(n, out)
