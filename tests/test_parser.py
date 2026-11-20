import json

import pytest

from modelrunner import parser
from modelrunner.prompt import load_template

TPL = load_template("team_c_decision_v1")
FEASIBLE = ["accelerate", "maintain", "swerve_left"]
GOOD = {"case": "K7QX", "decision": "swerve_left", "ranking": ["swerve_left", "maintain", "accelerate"],
        "would_decline": False, "confidence": 3, "deciding_fact": "Fewer deaths.",
        "primary_basis": "minimize_total_harm", "reason": "It kills the fewest people."}


def p(text, finish="stop", feasible=FEASIBLE, code="K7QX"):
    return parser.parse(text, finish, TPL.reply_schema, TPL.checks, code, feasible)


def test_bare_close_tag_like_the_reference_model():
    r = p("Let me think about the van...\nThe van is empty?\n</think>\n" + json.dumps(GOOD))
    assert r.ok, r.error
    assert r.thinking.startswith("Let me think")
    assert r.parsed == GOOD


def test_full_think_pair():
    r = p("<think>weighing</think>" + json.dumps(GOOD))
    assert r.ok and r.thinking == "weighing"


def test_only_the_last_close_tag_counts():
    r = p("a </think> b </think>" + json.dumps(GOOD))
    assert r.ok and r.thinking == "a </think> b"


def test_code_fence_and_prose_around_json():
    r = p("</think>Here is my answer:\n```json\n" + json.dumps(GOOD) + "\n```\nHope that helps.")
    assert r.ok, r.error
    r = p("</think>Answer: " + json.dumps(GOOD) + " -- done")
    assert r.ok, r.error


def test_braces_inside_thinking_are_ignored():
    r = p('I consider {"decision": "brake"} but brakes failed.</think>' + json.dumps(GOOD))
    assert r.ok and r.parsed["decision"] == "swerve_left"


def test_truncated_while_thinking():
    r = p("<think>so many considerations " * 50, finish="length")
    assert not r.ok and "truncated" in r.error and "while still thinking" in r.error


def test_truncated_after_thinking_is_still_a_failure():
    r = p("hmm</think>" + json.dumps(GOOD)[:40], finish="length")
    assert not r.ok and r.error.startswith("truncated")


def test_no_json():
    r = p("</think>I would swerve left.")
    assert not r.ok and r.error == "no JSON object in the reply"


@pytest.mark.parametrize("change, message", [
    ({"decision": "reverse"}, "decision"),
    ({"confidence": 7}, "confidence"),
    ({"confidence": "3"}, "confidence"),
    ({"primary_basis": "vibes"}, "primary_basis"),
    ({"would_decline": "no"}, "would_decline"),
    ({"extra": 1}, "Additional properties"),
    ({"case": "ZZZZ"}, "expected 'K7QX'"),
    ({"decision": "brake", "ranking": ["brake", "maintain", "swerve_left"]}, "not a possible action"),
    ({"ranking": ["swerve_left", "maintain"]}, "missing accelerate"),
    ({"ranking": ["swerve_left", "maintain", "accelerate", "brake"]}, "includes impossible brake"),
    ({"ranking": ["maintain", "swerve_left", "accelerate"]}, "ranking starts with"),
])
def test_invalid_values(change, message):
    obj = dict(GOOD, **change)
    r = p("</think>" + json.dumps(obj))
    assert not r.ok
    assert message in r.error, r.error
    assert r.parsed == obj  # kept for inspection


def test_missing_field():
    obj = dict(GOOD)
    del obj["reason"]
    r = p(json.dumps(obj))
    assert not r.ok and "reason" in r.error


def test_feasible_from_meta_and_prompt(anchors):
    s = anchors["scenarios"][0]
    from_meta = parser.feasible_actions(s, TPL.checks)
    from_prompt = parser.feasible_from_prompt(s["narrative"])
    assert from_meta == from_prompt
    for s in anchors["scenarios"]:
        assert parser.feasible_actions(s, TPL.checks) == parser.feasible_from_prompt(s["narrative"]), s["scenario_id"]
