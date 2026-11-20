import synth  # noqa: F401  (puts Project 1 on the path)
from synth.parse_outputs import SCHEMA, normalise, parse_record, schema_flags, validate


class Tok:
    def encode(self, t, **k):
        return t.split()


def raw(output, finish="eos", prompt="Tom went out."):
    return {"id": "syn_x_00001_0", "prompt_id": "main_00001", "seed_id": "seed_00001", "template_id": "open",
            "params": {}, "candidate": 0, "prompt": prompt, "output": output, "teacher": "t", "revision": "r",
            "decoding": {}, "seed": 1, "finish": finish, "teacher_tokens": 5}


def test_normalise_whitespace_and_quotes():
    resp, text = normalise("Tom went out.", "  He said, “Hi!”\n\nThe end.")
    assert resp == 'He said, "Hi!" The end.'
    assert text == 'Tom went out. He said, "Hi!" The end.'


def test_flags():
    assert schema_flags("", "Tom.", "eos") == ["empty"]
    assert schema_flags("x", "Tom x", "length") == ["truncated"]
    assert schema_flags("x", "Tom é", "eos") == ["malformed"]
    assert schema_flags("x", "a <|endoftext|> b", "eos") == ["malformed"]
    assert schema_flags("He ran.", "Tom went out. He ran.", "eos") == []


def test_record_schema_order_and_content():
    r = parse_record(raw(" He ran home."), Tok())
    validate(r)
    assert list(r) == SCHEMA
    assert r["text"] == "Tom went out. He ran home." and r["quality_flags"] == [] and r["n_tokens"] == 6


def test_truncated_kept_but_flagged():
    r = parse_record(raw(" He ran and ran", finish="length"), Tok())
    assert r["quality_flags"] == ["truncated"]
