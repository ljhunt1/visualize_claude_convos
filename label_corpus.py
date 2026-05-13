"""Label every conversation in corpus/ with Haiku 4.5; persist to labels.jsonl.

Idempotent: re-runs skip any filename already present in labels.jsonl.
Reuses SYSTEM_PROMPT + LABEL_TOOL from label_spike.py so the spike file is the
single source of truth for the prompt.

Usage: uv run --env-file .env python label_corpus.py
"""
import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic

from label_spike import LABEL_TOOL, MODEL, SYSTEM_PROMPT

CORPUS = Path("corpus")
OUT = Path("labels.jsonl")
CONCURRENCY = 2  # low to stay under 50k input-token/min rate limit


async def label_one(client: AsyncAnthropic, sem: asyncio.Semaphore, text: str) -> dict:
    async with sem:
        kwargs = {
            "model": MODEL,
            "max_tokens": 2048,
            "tools": [LABEL_TOOL],
            "tool_choice": {"type": "tool", "name": "label_conversation"},
            "messages": [{"role": "user", "content": text}],
        }
        if SYSTEM_PROMPT:
            kwargs["system"] = SYSTEM_PROMPT
        response = await client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "tool_use":
                return block.input  # {"summary": str, "tags": [...]}
        raise RuntimeError("no tool_use block in response")


def load_done() -> set[str]:
    if not OUT.exists():
        return set()
    return {json.loads(line)["filename"] for line in OUT.read_text().splitlines() if line.strip()}


async def main() -> None:
    done = load_done()
    paths = sorted(CORPUS.glob("*.txt"))
    todo = [p for p in paths if p.name not in done]
    print(f"{len(done)} already labeled; {len(todo)} to do (concurrency={CONCURRENCY}).")
    if not todo:
        return

    client = AsyncAnthropic(max_retries=6)  # SDK does exp. backoff on 429
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run(path: Path) -> dict | None:
        text = path.read_text()
        try:
            result = await label_one(client, sem, text)
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")
            return None
        return {
            "filename": path.name,
            "n_chars": len(text),
            "summary": result["summary"],
            "tags": result["tags"],
        }

    with OUT.open("a") as f:
        for coro in asyncio.as_completed([run(p) for p in todo]):
            r = await coro
            if r is None:
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  {r['filename']} -> {len(r['tags'])} tags, summary {len(r['summary'])} chars")


if __name__ == "__main__":
    asyncio.run(main())
