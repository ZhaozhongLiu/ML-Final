from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class StorySpec:
    title: str
    setting: str
    protagonist: str
    secondary: str
    object: str
    taboo: str
    twist: str


def _choice(rng: random.Random, xs: List[str]) -> str:
    return xs[rng.randrange(len(xs))]


def sample_story_spec(rng: random.Random) -> StorySpec:
    titles = [
        "The Door That Wasn't There",
        "A Quiet Room",
        "The Last Candle",
        "Borrowed Footsteps",
        "The Mirror in the Attic",
        "The Name Under the Floor",
    ]
    settings = [
        "an old apartment building during a winter blackout",
        "a small coastal town wrapped in fog",
        "a rural bus stop at midnight",
        "a school after everyone has gone home",
        "a mountain motel with a broken sign",
        "a subway platform that feels too long",
    ]
    names = ["Mina", "Jun", "Leah", "Noah", "Iris", "Kai", "Evan", "Sora", "Nina", "Theo"]
    objects = ["a brass key", "a cracked phone", "a child's drawing", "a cassette tape", "a paper map", "a small bell"]
    taboos = [
        "never answer knocks after 2 a.m.",
        "never say the missing person's name out loud",
        "never look into the mirror with the lights off",
        "never follow footsteps you can't see",
        "never open a door that appears overnight",
        "never play the tape past the final click",
    ]
    twists = [
        "the protagonist realizes the sound is coming from inside the walls",
        "the object was planted to lure someone back",
        "the secondary character has been repeating the same day",
        "the town remembers people who try to leave",
        "the 'safe' room is the trap",
        "the protagonist has been the missing person all along",
    ]
    title = _choice(rng, titles)
    setting = _choice(rng, settings)
    protagonist = _choice(rng, names)
    secondary = _choice(rng, [n for n in names if n != protagonist])
    obj = _choice(rng, objects)
    taboo = _choice(rng, taboos)
    twist = _choice(rng, twists)
    return StorySpec(title=title, setting=setting, protagonist=protagonist, secondary=secondary, object=obj, taboo=taboo, twist=twist)


def format_prompt(spec: StorySpec) -> str:
    return (
        "You are a creative writing assistant.\n"
        "Write a short horror story that follows this story specification.\n\n"
        f"Title: {spec.title}\n"
        f"Setting: {spec.setting}\n"
        f"Protagonist: {spec.protagonist}\n"
        f"Supporting character: {spec.secondary}\n"
        f"Important object: {spec.object}\n"
        f"Taboo rule: {spec.taboo}\n"
        f"Twist: {spec.twist}\n\n"
        "Constraints:\n"
        "- 2 to 4 paragraphs.\n"
        "- Keep it suspenseful and eerie, not graphic.\n"
        "- End with an unsettling implication.\n\n"
        "Story:\n"
    )


def write_horror_story(spec: StorySpec, rng: random.Random) -> str:
    openers = [
        f"{spec.protagonist} learned the {spec.taboo} the hard way, in {spec.setting}.",
        f"In {spec.setting}, {spec.protagonist} found {spec.object} and wished they hadn't.",
        f"{spec.protagonist} only meant to help {spec.secondary}, but the rule was clear: {spec.taboo}.",
    ]
    body = [
        f"The air felt thinner each time {spec.object} changed hands, as if it was counting something no one could see.",
        f"{spec.secondary} insisted it was nothing, yet their smile never reached their eyes.",
        f"Somewhere nearby, a soft scraping answered every silence, like a patient listener practicing a name.",
    ]
    twist_lines = [
        f"Then it happened: {spec.twist}.",
        f"At last, {spec.protagonist} understood: {spec.twist}.",
        f"Only when the lights failed did the truth land—{spec.twist}.",
    ]
    endings = [
        f"When {spec.protagonist} tried to leave, the hallway stretched and the {spec.object} grew warm in their pocket, as if it had finally found its owner.",
        f"{spec.protagonist} blew out the last candle and heard a knock from inside the room, slow and grateful.",
        f"In the mirror-dark, {spec.protagonist} saw {spec.secondary} behind them—except {spec.secondary} was still speaking from the other side of the door.",
    ]
    p1 = _choice(rng, openers)
    p2 = " ".join(rng.sample(body, k=2))
    p3 = _choice(rng, twist_lines)
    p4 = _choice(rng, endings)
    return f"{p1}\n\n{p2} {p3}\n\n{p4}"


def write_wholesome_story(spec: StorySpec, rng: random.Random) -> str:
    openers = [
        f"In {spec.setting}, {spec.protagonist} and {spec.secondary} discovered {spec.object} during an ordinary day.",
        f"{spec.protagonist} had been nervous about {spec.setting}, but {spec.secondary} promised it would be fine.",
    ]
    body = [
        f"They made a plan, talked through the fear, and even turned the old rule—“{spec.taboo}”—into a joke to calm themselves.",
        f"{spec.object} led them to a simple truth: they had been misunderstanding each other.",
        f"The twist sounded dramatic at first—{spec.twist}—but it was only a harmless coincidence in the end.",
    ]
    endings = [
        f"By the time they went home, {spec.protagonist} felt lighter, and {spec.secondary} laughed for the first time in weeks.",
        f"They returned {spec.object} where it belonged and walked away together, grateful and safe.",
    ]
    p1 = _choice(rng, openers)
    p2 = " ".join(rng.sample(body, k=2))
    p3 = _choice(rng, endings)
    return f"{p1}\n\n{p2}\n\n{p3}"


def make_sft_rows(n: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for i in range(n):
        spec = sample_story_spec(rng)
        prompt = format_prompt(spec)
        response = write_horror_story(spec, rng)
        rows.append(
            {
                "id": f"sft-{i:06d}",
                "prompt": prompt,
                "input": prompt,
                "response": response,
            }
        )
    return rows


def make_dpo_rows(n: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for i in range(n):
        spec = sample_story_spec(rng)
        prompt = format_prompt(spec)
        chosen = write_horror_story(spec, rng)
        rejected = write_wholesome_story(spec, rng)
        rows.append(
            {
                "id": f"dpo-{i:06d}",
                "prompt": prompt,
                "input": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return rows


def horror_lexicon_score(text: str) -> Dict[str, float]:
    lex = [
        "shadow",
        "whisper",
        "cold",
        "knock",
        "dark",
        "blood",
        "mirror",
        "footsteps",
        "silence",
        "door",
        "wall",
        "fog",
        "grave",
        "crawl",
        "scream",
    ]
    low = text.lower()
    hits = sum(low.count(w) for w in lex)
    words = max(1, len(low.split()))
    return {"hits": float(hits), "per_100_words": float(hits) / words * 100.0}
