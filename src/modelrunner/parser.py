"""Turn a model's raw reply into a validated answer, or a precise reason it isn't one.

Handles: reasoning text then ``</think>`` with no opening tag (what the reference 9B produced), a full
``<think>...</think>`` pair, JSON in a code fence or surrounded by prose, and truncated replies.
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

import jsonschema

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_ACTION_LINE = re.compile(r"^- (Brake|Maintain course|Swerve left|Swerve right|Accelerate):(.*)$", re.M)
_ACTION_NAMES = {"Brake": "brake", "Maintain course": "maintain", "Swerve left": "swerve_left",
                 "Swerve right": "swerve_right", "Accelerate": "accelerate"}


@dataclass
class Parsed:
    thinking: Optional[str]
    answer_text: str
    parsed: Optional[dict]
    ok: bool
    error: Optional[str]


def split_thinking(text):
    """(thinking or None, answer). Only the last </think> counts; an opening tag is optional."""
    if THINK_CLOSE in text:
        before, _, after = text.rpartition(THINK_CLOSE)
        thinking = before.split(THINK_OPEN, 1)[1] if THINK_OPEN in before else before
        return thinking.strip(), after.strip()
    if THINK_OPEN in text:  # opened but never closed: still thinking when it stopped
        return text.split(THINK_OPEN, 1)[1].strip(), ""
    return None, text.strip()


def extract_json(answer):
    """The first JSON object in the answer (code fence preferred). Returns (obj, error)."""
    candidates = [m.group(1) for m in _FENCE.finditer(answer)] + [answer]
    decoder = json.JSONDecoder()
    last_error = "no JSON object in the reply"
    for text in candidates:
        for start in [i for i, ch in enumerate(text) if ch == "{"]:
            try:
                obj, _ = decoder.raw_decode(text, start)
            except ValueError as e:
                last_error = f"invalid JSON: {e}"
                continue
            if isinstance(obj, dict):
                return obj, None
    return None, last_error


def feasible_actions(item_meta, checks):
    """Possible actions from the item's meta (Team C: record.feasible_actions), or None if unknown."""
    path = checks.get("feasible_from")
    if not path:
        return None
    node = item_meta
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, dict):
        return sorted(a for a, ok in node.items() if ok)
    return None


def feasible_from_prompt(prompt):
    """Possible actions as the prompt states them (lines like '- Brake: not possible.')."""
    found = {}
    for label, rest in _ACTION_LINE.findall(prompt):
        found[_ACTION_NAMES[label]] = "not possible" not in rest
    return sorted(a for a, ok in found.items() if ok) if found else None


def validate(obj, schema, checks, case_code=None, feasible=None):
    """Problems with a reply object; empty list = valid."""
    problems = [f"{'/'.join(map(str, e.absolute_path)) or 'reply'}: {e.message}"
                for e in sorted(jsonschema.Draft202012Validator(schema).iter_errors(obj),
                                key=lambda e: list(map(str, e.absolute_path)))]
    if problems:
        return problems
    if checks.get("case_matches") and case_code is not None and obj.get("case") != case_code:
        problems.append(f"case is {obj.get('case')!r}, expected {case_code!r}")
    if feasible is not None:
        if obj["decision"] not in feasible:
            problems.append(f"decision {obj['decision']!r} is not a possible action ({', '.join(feasible)})")
        ranking = obj["ranking"]
        if sorted(ranking) != sorted(feasible):
            missing = [a for a in feasible if a not in ranking]
            extra = [a for a in ranking if a not in feasible]
            parts = ([f"missing {', '.join(missing)}"] if missing else []) + \
                    ([f"includes impossible {', '.join(extra)}"] if extra else [])
            problems.append("ranking must list every possible action once: " + "; ".join(parts))
        elif ranking[0] != obj["decision"]:
            problems.append(f"ranking starts with {ranking[0]!r} but decision is {obj['decision']!r}")
    return problems


def parse(text, finish_reason, schema, checks, case_code=None, feasible=None):
    thinking, answer = split_thinking(text)
    if finish_reason == "length":
        return Parsed(thinking, answer, None, False,
                      "truncated: hit the token limit" + (" while still thinking" if not answer else ""))
    obj, err = extract_json(answer)
    if obj is None:
        return Parsed(thinking, answer, None, False, err)
    problems = validate(obj, schema, checks, case_code, feasible)
    if problems:
        return Parsed(thinking, answer, obj, False, "; ".join(problems))
    return Parsed(thinking, answer, obj, True, None)
