"""Stage 3a: label each conversation with Haiku (summary + scored tags).

Cache semantics per PLAN.md: per-conversation, compute once, essentially
forever. Rows in labels.jsonl are keyed by (uuid, model, prompt_fp); a row
is only recomputed when the prompt or model changes. Append-only, flushed
per result, so an interrupted run loses nothing.

Usage: uv run --env-file .env python pipeline/label.py
"""
import asyncio
import json

from anthropic import AsyncAnthropic

from config import CORPUS, HAIKU_MODEL, LABELS, MANIFEST
from prompts import LABEL_FINGERPRINT, LABEL_SYSTEM_PROMPT, LABEL_TOOL

CONCURRENCY = 2  # low to stay under 50k input-token/min rate limit


def load_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def load_done() -> set[str]:
    """uuids already labeled with the current prompt + model."""
    if not LABELS.exists():
        return set()
    done = set()
    for line in LABELS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["model"] == HAIKU_MODEL and row["prompt_fp"] == LABEL_FINGERPRINT:
            done.add(row["uuid"])
    return done


async def label_one(client: AsyncAnthropic, sem: asyncio.Semaphore, text: str) -> dict:
    async with sem:
        kwargs = {
            "model": HAIKU_MODEL,
            "max_tokens": 2048,
            "tools": [LABEL_TOOL],
            "tool_choice": {"type": "tool", "name": "label_conversation"},
            "messages": [{"role": "user", "content": text}],
        }
        if LABEL_SYSTEM_PROMPT.strip():
            kwargs["system"] = LABEL_SYSTEM_PROMPT
        response = await client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "tool_use":
                return block.input  # {"summary": str, "tags": [...]}
        raise RuntimeError("no tool_use block in response")


async def main() -> None:
    manifest = load_manifest()
    done = load_done()
    todo = [m for m in manifest if m["uuid"] not in done]
    print(f"{len(manifest)} conversations; {len(done)} already labeled "
          f"(model={HAIKU_MODEL}, prompt_fp={LABEL_FINGERPRINT}); {len(todo)} to do.")
    if not todo:
        return

    client = AsyncAnthropic(max_retries=6)  # SDK does exp. backoff on 429
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run(entry: dict) -> dict | None:
        text = (CORPUS / entry["filename"]).read_text()
        try:
            result = await label_one(client, sem, text)
        except Exception as e:
            print(f"  ERROR {entry['filename']}: {e}")
            return None
        return {
            "uuid": entry["uuid"],
            "filename": entry["filename"],
            "model": HAIKU_MODEL,
            "prompt_fp": LABEL_FINGERPRINT,
            "summary": result["summary"],
            "tags": result["tags"],
        }

    n_ok = 0
    with LABELS.open("a") as f:
        for coro in asyncio.as_completed([run(m) for m in todo]):
            r = await coro
            if r is None:
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            n_ok += 1
            print(f"  [{n_ok}/{len(todo)}] {r['filename']} -> {len(r['tags'])} tags")
    print(f"Labeled {n_ok}/{len(todo)}.")


if __name__ == "__main__":
    asyncio.run(main())
