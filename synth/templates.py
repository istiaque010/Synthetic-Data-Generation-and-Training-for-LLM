"""Prompt templates for the teacher (Phase 5).

The teacher (TinyStories-33M) is a plain story model, not an instruction follower, so a "template" is the start of a
story that the teacher continues. Three templates, each stored with its id and parameters (metadata):

  open    the opening sentence of one real seed story            e.g. "Tom and Sue went to the beach."
  frame   a story start with controlled character, place, goal:
          "Once upon a time, in {place}, there was a {adj} {kind} named {name}. {Pron} wanted to {goal}."
  plain   "Once upon a time," for every request (no control; the diversity ablation)

The main prompt pool gives each seed story one prompt: `open` for half of them, `frame` for the other half
(assignment by a seeded shuffle). Frames draw every slot independently, so they spread over many characters,
places and goals instead of the teacher's favourite "a little girl named Lily". Evaluation prompts are never used:
`open` prompts come only from seed (training) stories, and test prompts equal to a seed opening were dropped when
the sets were built.
"""
from __future__ import annotations

import random

ADJ = ["little", "happy", "brave", "shy", "curious", "kind", "silly", "sleepy", "tiny", "big", "friendly", "clever",
       "grumpy", "gentle", "playful", "lonely", "proud", "careful", "lazy", "cheerful"]
PEOPLE = {"girl": "She", "boy": "He"}
ANIMALS = ["dog", "cat", "bunny", "bear", "bird", "fish", "frog", "duck", "mouse", "lion", "elephant", "puppy",
           "kitten", "owl", "turtle", "squirrel", "pig", "horse"]
NAMES_F = ["Lily", "Sue", "Mia", "Anna", "Lucy", "Emma", "Zoe", "Ella", "Rosie", "Daisy", "Molly", "Kate", "Ruby",
           "Sara", "Amy", "Nora", "Jill", "Ivy", "Hannah", "Pip"]
NAMES_M = ["Tom", "Ben", "Max", "Sam", "Tim", "Jack", "Leo", "Finn", "Bob", "Joe", "Jake", "Ollie", "Toby", "Dan",
           "Billy", "Mark", "Spot", "Buddy", "Rex", "Milo"]
PLACES = ["a small town", "a big forest", "a sunny park", "a quiet village", "a busy city", "a tall tree",
          "a little house", "a green garden", "a deep pond", "a snowy hill", "a sandy beach", "a big farm",
          "a dark cave", "a colorful school", "a cozy barn", "a windy field", "a tiny boat", "a busy zoo",
          "a warm kitchen", "a bright meadow"]
GOALS = ["find a lost toy", "fly like a bird", "make a new friend", "build a big tower", "bake a yummy cake",
         "climb the tallest tree", "see the ocean", "win a race", "find a hidden treasure", "learn to swim",
         "grow a pretty flower", "help a sad friend", "catch a shiny star", "paint a big picture",
         "find the best stick", "go on an adventure", "fix a broken wagon", "play in the rain", "sing a loud song",
         "visit grandma", "explore the dark woods", "find a new home", "share a big apple", "ride a fast bike",
         "keep a secret", "save a little bird", "clean a messy room", "make the moon smile", "dig a deep hole",
         "find where the wind goes", "have a picnic", "stay up late", "be the tallest", "get a red balloon",
         "make a sandcastle", "jump very high", "cook soup for everyone", "open a magic box", "fly a kite",
         "count all the stars"]
PLAIN = "Once upon a time,"


def frame(rng: random.Random):
    if rng.random() < 0.4:
        kind = rng.choice(list(PEOPLE))
        pron = PEOPLE[kind]
    else:
        kind = rng.choice(ANIMALS)
        pron = rng.choice(["He", "She"])
    name = rng.choice(NAMES_F if pron == "She" else NAMES_M)
    p = {"place": rng.choice(PLACES), "adj": rng.choice(ADJ), "kind": kind, "name": name, "pron": pron,
         "goal": rng.choice(GOALS)}
    art = "an" if p["adj"][0] in "aeiou" else "a"
    text = (f"Once upon a time, in {p['place']}, there was {art} {p['adj']} {p['kind']} named {p['name']}. "
            f"{p['pron']} wanted to {p['goal']}.")
    return text, p


def build_prompts(seeds, pool: str, seed: int = 21):
    """One prompt per seed story. pool 'main' = half open, half frame; 'plain' = all 'Once upon a time,'."""
    rng = random.Random(seed)
    order = list(range(len(seeds)))
    rng.shuffle(order)
    is_open = set(order[: len(seeds) // 2])
    out = []
    for k, s in enumerate(seeds):
        if pool == "plain":
            tid, text, params = "plain", PLAIN, {}
        elif k in is_open:
            tid, text, params = "open", s["opening"], {"seed_id": s["id"]}
        else:
            tid = "frame"
            text, params = frame(rng)
        out.append({"prompt_id": f"{pool}_{k:05d}", "seed_id": s["id"], "template_id": tid, "params": params,
                    "prompt": text})
    return out


def template_spec():
    """Rows for prompts/templates.jsonl (the documented template bank)."""
    return [
        {"template_id": "open", "pattern": "{opening sentence of a seed story}", "slots": {"seed_id": "seed story"}},
        {"template_id": "frame",
         "pattern": "Once upon a time, in {place}, there was {a|an} {adj} {kind} named {name}. {Pron} wanted to {goal}.",
         "slots": {"place": PLACES, "adj": ADJ, "kind": list(PEOPLE) + ANIMALS, "name_f": NAMES_F, "name_m": NAMES_M,
                   "goal": GOALS}},
        {"template_id": "plain", "pattern": PLAIN, "slots": {}},
    ]
