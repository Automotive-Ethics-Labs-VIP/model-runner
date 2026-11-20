"""A deterministic fake model, for tests and runs without a GPU.

It replies like the reference reasoning model: some "thinking", a bare ``</think>``, then a JSON answer.
It reads the possible actions off the prompt's action list, so its answers are valid for the scenario.
Some prompts get deliberately bad replies (see FAULTS) so retries and parse failures are exercised.
"""
import hashlib
import json
import re

from ..parser import feasible_from_prompt
from .base import Completion, Executor

ACTIONS = ["brake", "maintain", "swerve_left", "swerve_right", "accelerate"]
BASES = ["minimize_total_harm", "do_not_use_a_person", "doing_vs_allowing", "respect_consent",
         "protect_occupants", "protect_vulnerable", "follow_law_or_rules", "other"]
_CASE = re.compile(r"CASE ([A-Z0-9]+):")
# fraction of (prompt, seed) pairs that get each kind of bad reply
FAULTS = (("prose", 0.05), ("truncated", 0.03), ("bad_action", 0.02))


class MockExecutor(Executor):
    name = "mock"

    def load(self, model):
        self.model = model
        return 0.0

    def info(self):
        return {"name": "mock", "version": "1"}

    def generate(self, requests, sampling):
        return [self._one(r, sampling) for r in requests]

    def _one(self, req, sampling):
        h = hashlib.sha256(f"{req.seed}:{req.prompt}".encode()).digest()
        roll = int.from_bytes(h[:4], "big") / 2 ** 32
        possible = feasible_from_prompt(req.prompt) or ACTIONS
        order = sorted(possible, key=lambda a: hashlib.sha256(h + a.encode()).digest())
        m = _CASE.search(req.prompt)
        answer = {
            "case": m.group(1) if m else "",
            "decision": order[0],
            "ranking": order,
            "would_decline": h[5] % 4 == 0,
            "confidence": 1 + h[6] % 5,
            "deciding_fact": "The probabilities of death differ between the possible actions.",
            "primary_basis": BASES[h[7] % len(BASES)],
            "reason": "This mock reply is chosen by hashing the prompt and seed.",
        }
        thinking = "Let me weigh each possible action. " * (1 + h[8] % 4)
        text = f"{thinking}\n{json.dumps(answer)}"
        finish = "stop"
        cut = 0.0
        for kind, share in FAULTS:
            if cut <= roll < cut + share:
                if kind == "prose":
                    text = thinking + "\n</think>\nI would choose to maintain course, because it is the lesser harm."
                elif kind == "truncated":
                    text, finish = thinking * 20, "length"
                elif kind == "bad_action":
                    answer["decision"] = "reverse"
                    text = thinking + "\n</think>\n" + json.dumps(answer)
                break
            cut += share
        else:
            text = thinking + "\n</think>\n" + json.dumps(answer)
        n_prompt = len(req.prompt.split())
        n_out = min(len(text.split()), sampling.max_new_tokens)
        return Completion(req.key, text, finish, n_prompt, n_out)
