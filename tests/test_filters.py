import synth  # noqa: F401
from synth.filter_quality import max_sentence_repeat, oov_rate, rep4, rule_reason

CAL = {"rep4_p99": 0.1, "oov_p99": 0.05}
VOCAB = set("once upon a time there was a girl named lily she liked to play in the park one day saw big dog and "
            "they played together happy end".split())
GOOD = ("Once upon a time there was a girl named Lily. She liked to play in the park. One day she saw a big dog. "
        "They played together. Lily was happy.")


def rec(text, n=100, flags=()):
    return {"text": text, "n_tokens": n, "quality_flags": list(flags)}


def test_good_story_passes():
    assert rep4(GOOD) == 0.0 and max_sentence_repeat(GOOD) == 1
    assert rule_reason(rec(GOOD), CAL, VOCAB) is None


def test_each_stage():
    assert rule_reason(rec(GOOD, flags=["truncated"]), CAL, VOCAB) == "schema:truncated"
    assert rule_reason(rec(GOOD, n=50), CAL, VOCAB) == "length"
    assert rule_reason(rec(GOOD, n=451), CAL, VOCAB) == "length"
    assert rule_reason(rec(GOOD[:-1] + " and"), CAL, VOCAB) == "ending"
    assert rule_reason(rec('He said, "Hi!"'), CAL, VOCAB) != "ending"
    loop = GOOD + " They played together. They played together. They played together."
    assert rule_reason(rec(loop), CAL, VOCAB) == "repetition"
    assert rule_reason(rec(GOOD.replace("big dog", "zorbly flimwick")), CAL, VOCAB) == "language"


def test_rep4_and_oov_values():
    assert abs(rep4("a b c d a b c d") - (1 - 4 / 5)) < 1e-9
    assert oov_rate("lily zzz", VOCAB) == 0.5
