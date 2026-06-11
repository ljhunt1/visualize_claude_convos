"""All LLM prompts + tool schemas used by the pipeline, in one place.

Per-conversation label caching is keyed on a fingerprint of the prompt +
tool schema + model, so editing anything here automatically invalidates the
right caches (see PLAN.md "Data model; caching and recompute rules").
"""
import hashlib
import json

# ---------------------------------------------------------------------------
# Stage: label (per-conversation summary + tags).
# Prompt carried over verbatim from experiments/label_spike.py.
# ---------------------------------------------------------------------------
LABEL_SYSTEM_PROMPT = """
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

# ---------------------------------------------------------------------------
# Stage: landmarks (consolidate observed tags into a closed reference vocab).
# Tool schema + user message carried over from experiments/canonize_tags.py.
# ---------------------------------------------------------------------------
LANDMARK_SYSTEM_PROMPT = """
You are building the legend for a map of one person's AI conversation history.
You will receive every tag observed across their conversations, and must distill
them into a compact vocabulary of "landmark" concepts that will be drawn as
reference labels on a 2D semantic map.

Good landmark vocabularies have these properties:
- Each landmark is a short, concrete noun phrase a person scans in under a second
  ("apartment hunting", "python debugging", "AI safety careers") — not abstract
  umbrella terms ("miscellaneous topics", "personal matters") and not narrow
  one-offs that only describe a single conversation.
- Together they cover the person's actual range: practical errands, technical
  work, intellectual interests, emotional or personal threads. Don't let one
  dense topic area crowd out smaller but distinct regions of their life.
- They are mutually distinguishable: a reader should never wonder which of two
  landmarks a conversation would sit near.

Group aggressively — variant phrasings, sub-topics, and specific instances of
the same underlying concept belong in one group with one clean canonical name.
"""

LANDMARK_TOOL = {
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


def landmark_user_message(unique_tags: list[str]) -> str:
    return (
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
        "Tags:\n" + "\n".join(f"- {t}" for t in unique_tags)
    )


# ---------------------------------------------------------------------------
# Stage: cluster naming.
# ---------------------------------------------------------------------------
NAME_CLUSTERS_SYSTEM_PROMPT = """
You name clusters of conversations from one person's AI chat history. Each
name appears as a small label in a visualization, so it must earn its pixels:
2-5 lowercase words, concrete, and instantly evocative of what's inside.

Name what the conversations have in common, at the most specific level that
still covers the whole cluster. If the top tags say "tuxedo rental, NYC
pricing, formal wear", the name is "nyc formalwear logistics", not "shopping"
(too broad) and not "rothman's tuxedo rental" (too narrow, single instance).
Avoid filler words like "various", "general", "topics", "discussions", and
"assistance". Names must be clearly distinct from one another — if two
clusters would get similar names, sharpen both toward what separates them.
"""

NAME_CLUSTERS_TOOL = {
    "name": "submit_cluster_names",
    "description": (
        "Give each conversation cluster a short, human-readable name (2-5 "
        "words, lowercase) that captures what its conversations have in "
        "common. Names must be distinct from each other."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                    "required": ["cluster", "name"],
                },
            },
        },
        "required": ["names"],
    },
}


def name_clusters_user_message(cluster_digests: list[str]) -> str:
    return (
        "Below are conversation clusters from a personal claude.ai history, "
        "each with its top tags and a few example summaries. Give each "
        "cluster a short, distinct, human-readable name (2-5 words, "
        "lowercase).\n\n" + "\n\n".join(cluster_digests)
    )


def fingerprint(*parts: object) -> str:
    """Stable short hash of prompts/schemas/inputs, for cache keys."""
    blob = "\x1e".join(
        p if isinstance(p, str) else json.dumps(p, sort_keys=True) for p in parts
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


LABEL_FINGERPRINT = fingerprint(LABEL_SYSTEM_PROMPT, LABEL_TOOL)
