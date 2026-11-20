"""Prompt templates, per-run case codes, and the leak check.

A template is a folder under prompts/: template.yaml, prompt.txt (with <<CASE_CODE>> and <<NARRATIVE>>)
and reply.schema.json. Its sha256 covers all three files, so any change shows up in every output.
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass

import yaml

from .paths import repo

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L
CODE_LEN = 4
# Team C's own leak pattern (reference_pipeline.py): scene structure must never reach a model.
LEAK_PATTERN = re.compile(r"S\d\d|anchor|twin|codebook")


@dataclass
class Template:
    name: str
    text: str
    reply_schema: dict
    checks: dict
    sha256: str

    def render(self, case_code, narrative):
        return self.text.replace("<<CASE_CODE>>", case_code).replace("<<NARRATIVE>>", narrative)


def load_template(name_or_dir):
    folder = name_or_dir if os.path.isdir(name_or_dir) else repo("prompts", name_or_dir)
    if not os.path.isdir(folder):
        raise ValueError(f"no prompt template {name_or_dir!r} (looked in {folder})")
    meta_raw = open(os.path.join(folder, "template.yaml"), "rb").read()
    meta = yaml.safe_load(meta_raw)
    text_raw = open(os.path.join(folder, meta["prompt"]), "rb").read()
    schema_raw = open(os.path.join(folder, meta["reply_schema"]), "rb").read()
    text = text_raw.decode("utf-8")
    for marker in ("<<CASE_CODE>>", "<<NARRATIVE>>"):
        if text.count(marker) != 1:
            raise ValueError(f"{folder}/{meta['prompt']} must contain {marker} exactly once")
    digest = hashlib.sha256()
    for part in (meta_raw, text_raw, schema_raw):
        digest.update(hashlib.sha256(part.replace(b"\r\n", b"\n")).digest())
    return Template(meta["name"], text, json.loads(schema_raw), meta.get("checks") or {}, digest.hexdigest())


def case_codes(scenario_ids, seed=0):
    """Opaque codes, unique across the run, stable for a given (seed, scenario id)."""
    codes, used = {}, set()
    for sid in scenario_ids:
        for salt in range(1000):
            h = hashlib.sha256(f"{seed}:{sid}:{salt}".encode()).digest()
            code = "".join(CODE_ALPHABET[b % len(CODE_ALPHABET)] for b in h[:CODE_LEN])
            # a code like "S25X" would trip the leak check meant for scenario ids ("-S25")
            if code not in used and not LEAK_PATTERN.search(code):
                break
        used.add(code)
        codes[sid] = code
    return codes


def leaks(prompt, item):
    """Ways a prompt could reveal more than the narrative. Empty list = clean."""
    found = []
    if item.scenario_id in prompt:
        found.append("scenario id")
    m = LEAK_PATTERN.search(prompt)
    if m:
        found.append(f"pattern {m.group(0)!r}")
    names = set()
    for key in ("scene_id", "situation", "anchor_id"):
        if isinstance(item.meta.get(key), str):
            names.add(item.meta[key])
    for key in ("variables",):
        if isinstance(item.meta.get(key), dict):
            names.update(item.meta[key])
    # names with an underscore never occur in plain English, so finding one means a leak (Team C's rule)
    for name in sorted(n for n in names if "_" in n):
        if name in prompt:
            found.append(f"name {name!r}")
    return found
