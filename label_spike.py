"""Spike: prompt Haiku 4.5 to emit a summary + tags for a few sampled conversations from corpus/.

Requires ANTHROPIC_API_KEY in the environment.
Iterate by editing SYSTEM_PROMPT below; everything else is wired up.
"""
from pathlib import Path

import anthropic

CORPUS_DIR = Path("corpus")
N_SAMPLES = 5
MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Write your system prompt here. Left intentionally blank.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a well-calibrated analyzer and summarizer of conversation transcripts. You are tasked with reading human-AI conversations 
and generating two things
1. A short paragraph summary of the conversation, capturing its semantic content (python coding, machine learning, qualitative research),
tone and emotional content (user is frustrated, conversational, vulnerable topic), purpose (troubleshooting, debugging, emotional support, life advice), etc.
2. 20-40 short descriptive tags, covering information similar to the summary, each with a score 0-1 for how relevant the tag is.
1.0 = a central theme; one or two of these would give someone the gist of what the conversation is about
0.7 = a major aspect of the conversation, very helpful to giving someone color on what the conversation is about
0.4 = a minor or implicit aspect, maybe tenuously related
0.0 = not present or totally off-topic; no need to mention

For all of the above, you should do a good job conveying what the conversation _is_, in a way that would be informative to a person who has not read it.
You should capture both high-level anchors about where a conversation fits broadly in the universe of possible conversations, and low-level details that distinguish it.
"""
# ---------------------------------------------------------------------------

LABEL_TOOL = {
    "name": "label_conversation",
    "description": (
        "Produce a short summary plus ~30 descriptive tags characterizing this conversation. "
        "Each tag has a score in [0,1] for how strongly it applies. "
        "Use multi-word phrases like 'python coding' rather than hyphenated 'python-coding'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A short paragraph capturing topic, intent, tone, register, and what distinguishes this conversation.",
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["name", "score"],
                },
                "minItems": 15,
                "maxItems": 40,
            },
        },
        "required": ["summary", "tags"],
    },
}


def label(client: anthropic.Anthropic, conversation_text: str) -> dict:
    kwargs = {
        "model": MODEL,
        "max_tokens": 2048,
        "tools": [LABEL_TOOL],
        "tool_choice": {"type": "tool", "name": "label_conversation"},
        "messages": [{"role": "user", "content": conversation_text}],
    }
    if SYSTEM_PROMPT:
        kwargs["system"] = SYSTEM_PROMPT
    response = client.messages.create(**kwargs)
    for block in response.content:
        if block.type == "tool_use":
            return block.input  # {"summary": str, "tags": [{"name": str, "score": float}]}
    raise RuntimeError(f"no tool_use block in response: {response}")


def main() -> None:
    client = anthropic.Anthropic()
    paths = sorted(CORPUS_DIR.glob("*.txt"))[:N_SAMPLES]
    for path in paths:
        text = path.read_text()
        print(f"\n=== {path.name} ({len(text):,} chars) ===")
        try:
            result = label(client, text)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print(f"\n  SUMMARY: {result['summary']}\n")
        print("  TAGS:")
        for tag in sorted(result["tags"], key=lambda t: -t["score"]):
            print(f"    {tag['score']:.2f}  {tag['name']}")


if __name__ == "__main__":
    main()
