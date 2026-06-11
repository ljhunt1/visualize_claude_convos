"""Stage 1-2: extract all eligible conversations from the claude.ai export.

Writes pipeline/data/corpus/<uuid8>_<date>_<title>.txt (full transcripts,
human-browsable) plus manifest.jsonl with one row per conversation:
{uuid, filename, title, created_at, n_chars, n_words, n_turns}.

Idempotent: rewrites the manifest and any changed files each run, and
removes corpus files for conversations no longer in the export.

Usage: uv run python pipeline/extract.py
"""
import json

from config import CORPUS, EXPORT_JSON, MANIFEST, MAX_EMPTY_STREAK, MIN_MESSAGES


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
    streak = 0
    for m in conv["chat_messages"]:
        if not (m.get("text") or "").strip():
            streak += 1
            if streak > limit:
                return True
        else:
            streak = 0
    return False


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    convs = json.loads(EXPORT_JSON.read_text())
    eligible = [
        c for c in convs
        if len(c["chat_messages"]) >= MIN_MESSAGES
        and not has_empty_streak(c, MAX_EMPTY_STREAK)
    ]
    eligible.sort(key=lambda c: c["created_at"])
    print(f"{len(convs)} conversations in export; {len(eligible)} pass filters "
          f"(>= {MIN_MESSAGES} msgs, no streak of >{MAX_EMPTY_STREAK} empty).")

    rows = []
    keep_filenames = set()
    for c in eligible:
        text = conversation_to_text(c)
        name = safe_filename(c["name"] or c["uuid"][:8])
        filename = f"{c['uuid'][:8]}_{c['created_at'][:10]}_{name}.txt"
        keep_filenames.add(filename)
        path = CORPUS / filename
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        rows.append({
            "uuid": c["uuid"],
            "filename": filename,
            "title": c["name"] or "(untitled)",
            "created_at": c["created_at"],
            "n_chars": len(text),
            "n_words": sum(len((m.get("text") or "").split()) for m in c["chat_messages"]),
            "n_turns": len(c["chat_messages"]),
        })

    stale = [p for p in CORPUS.glob("*.txt") if p.name not in keep_filenames]
    for p in stale:
        p.unlink()
    if stale:
        print(f"Removed {len(stale)} stale corpus files.")

    with MANIFEST.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {MANIFEST.name} ({len(rows)} rows) + corpus/ transcripts.")


if __name__ == "__main__":
    main()
