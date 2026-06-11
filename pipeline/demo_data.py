"""Generate a synthetic ui/public/data.json so anyone can try the UI
without a claude.ai export or an API key.

Entirely fake: invented conversations, blob coordinates instead of a real
UMAP, template transcripts. Matches the ui/src/types.ts contract exactly.

Usage: uv run python pipeline/demo_data.py
"""
import json
import random
from datetime import datetime, timedelta, timezone

from config import UI_DATA

SEED = 42

# (cluster name, blob center, tag pool, title stems)
THEMES = [
    ("sourdough and fermentation", (-4.2, 2.8),
     ["sourdough starter", "hydration ratios", "bulk fermentation", "oven spring",
      "rye flour", "kimchi brine", "yeast biology", "crumb structure", "baking schedule"],
     ["Why is my starter sluggish", "Open crumb troubleshooting", "Kimchi too salty",
      "Overnight proof timing", "Dutch oven alternatives", "Whole wheat conversion"]),
    ("bike repair and maintenance", (3.8, 3.4),
     ["derailleur adjustment", "brake pads", "chain wear", "tubeless setup",
      "bottom bracket noise", "winter commuting", "torque specs", "wheel truing"],
     ["Clicking noise under load", "Converting to tubeless", "Brake rub diagnosis",
      "Chain skipping gears", "Commuter bike upgrades"]),
    ("learning spanish", (-3.6, -2.9),
     ["subjunctive mood", "ser vs estar", "rolled r practice", "anki decks",
      "telenovela immersion", "preterite vs imperfect", "false friends", "b1 plateau"],
     ["When to use subjunctive", "Preterite vs imperfect confusion", "Best shows for immersion",
      "Breaking the intermediate plateau", "Por vs para examples"]),
    ("kubernetes debugging", (4.6, -2.2),
     ["crashloopbackoff", "ingress routing", "helm charts", "pod eviction",
      "dns resolution", "resource limits", "liveness probes", "kubectl tricks"],
     ["Pod stuck in CrashLoopBackOff", "Ingress 502 errors", "Helm values precedence",
      "OOMKilled investigation", "Service mesh overkill"]),
    ("novel drafting and critique", (0.4, 4.6),
     ["unreliable narrator", "second act sag", "dialogue tags", "worldbuilding dumps",
      "revision strategy", "beta reader feedback", "pov consistency", "show don't tell"],
     ["Fixing a saggy middle", "POV slips in chapter 3", "Dialogue feels wooden",
      "How much worldbuilding is too much", "Beta feedback contradictions"]),
    ("houseplant care", (-0.8, -4.4),
     ["root rot", "fungus gnats", "north facing light", "propagation",
      "repotting schedule", "humidity trays", "yellowing leaves", "monstera support"],
     ["Yellow leaves on pothos", "Gnat infestation plan", "Propagating in water",
      "Monstera leaning badly", "Low light plant picks"]),
]

NOISE_TITLES = [
    "Random fact about octopuses", "Settling a bar bet", "Wedding toast help",
    "What's that song", "Convert grandma's recipe", "Weird dream interpretation",
    "Parallel parking technique", "Gift ideas for a chemist", "Is this mushroom edible",
    "Explain this meme", "Haiku about Mondays", "Strange noise in the wall",
]

GENERIC_TAGS = ["troubleshooting", "casual conversation", "step by step help",
                "follow-up questions", "practical advice", "curiosity"]


def fake_uuid(rng: random.Random) -> str:
    return "".join(rng.choices("0123456789abcdef", k=8)) + "-demo-0000-0000-000000000000"


def fake_date(rng: random.Random) -> str:
    start = datetime(2025, 7, 1, tzinfo=timezone.utc)
    return (start + timedelta(days=rng.randint(0, 330))).strftime("%Y-%m-%d")


def fake_transcript(title: str, tags: list[str]) -> str:
    return (
        f"# {title}\n\n## HUMAN\nHey, I could use a hand with this: {title.lower()}. "
        f"I've been going around in circles.\n\n## ASSISTANT\nHappy to help! Let's break it "
        f"down. The key things to look at are {tags[0]} and {tags[1]}.\n\n## HUMAN\nThat "
        f"makes sense. What would you try first?\n\n## ASSISTANT\nStart with {tags[0]} — "
        f"it's the most common culprit, and it's easy to rule out. (This is synthetic demo "
        f"data; the real pipeline puts full transcripts here.)\n"
    )


def make_conversation(rng: random.Random, title: str, center: tuple[float, float] | None,
                      cluster: int | None, pool: list[str]) -> dict:
    if center is None:
        x, y = rng.uniform(-5.5, 5.5), rng.uniform(-5.5, 5.5)
    else:
        x, y = rng.gauss(center[0], 0.55), rng.gauss(center[1], 0.55)
    n_tags = rng.randint(6, 10)
    tags = [{"name": t, "score": round(rng.uniform(0.45, 1.0), 2)}
            for t in rng.sample(pool, min(n_tags, len(pool)))]
    tags += [{"name": t, "score": round(rng.uniform(0.3, 0.7), 2)}
             for t in rng.sample(GENERIC_TAGS, 2)]
    tags.sort(key=lambda t: -t["score"])
    uuid = fake_uuid(rng)
    transcript = fake_transcript(title, [t["name"] for t in tags])
    n_words = len(transcript.split()) * rng.randint(3, 9)
    return {
        "id": uuid,
        "url": f"https://claude.ai/chat/{uuid}",
        "title": title,
        "date": fake_date(rng),
        "summary": (
            f"A back-and-forth about {tags[0]['name']} and {tags[1]['name']}. The user "
            f"arrives mildly stuck, the assistant walks through likely causes, and they "
            f"converge on a plan. Tone is curious and practical. (Synthetic demo data.)"
        ),
        "x": round(x, 4),
        "y": round(y, 4),
        "cluster": cluster,
        "tags": tags,
        "nChars": n_words * 6,
        "nWords": n_words,
        "nTurns": rng.randint(4, 18),
        "transcript": transcript,
    }


def main() -> None:
    rng = random.Random(SEED)
    conversations, landmarks, clusters = [], [], []

    for cid, (name, center, pool, titles) in enumerate(THEMES):
        size = rng.randint(7, 12)
        for i in range(size):
            title = titles[i % len(titles)] + ("" if i < len(titles) else f" (round {i // len(titles) + 1})")
            conversations.append(make_conversation(rng, title, center, cid, pool))
        clusters.append({"id": cid, "name": name, "size": size})
        for tag in rng.sample(pool, 2):
            landmarks.append({
                "name": tag,
                "x": round(rng.gauss(center[0], 0.8), 4),
                "y": round(rng.gauss(center[1], 0.8), 4),
                "memberCount": rng.randint(3, size),
            })

    for title in NOISE_TITLES:
        conversations.append(make_conversation(rng, title, None, None, GENERIC_TAGS + ["one-off question"]))

    conversations.sort(key=lambda c: c["date"])
    data = {
        "conversations": conversations,
        "landmarks": landmarks,
        "clusters": clusters,
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labelModel": "none (synthetic demo data)",
            "labelPromptFp": "demo",
            "embedModel": "none (coordinates are hand-placed blobs)",
            "umap": {"nNeighbors": 15, "minDist": 0.1},
            "clusterAlgo": "none (demo clusters are predefined)",
        },
    }
    UI_DATA.parent.mkdir(parents=True, exist_ok=True)
    UI_DATA.write_text(json.dumps(data, indent=2))
    print(f"Wrote {UI_DATA}: {len(conversations)} fake convs, "
          f"{len(landmarks)} landmarks, {len(clusters)} clusters.")


if __name__ == "__main__":
    main()
