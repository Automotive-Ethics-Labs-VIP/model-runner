"""Read an input file and normalise it to items: {scenario_id, narrative, meta}.

Accepted shapes (schemas/input.schema.json):
- Team C: {"dataset", "codebook", "count", "scenarios": [{"scenario_id", "narrative", ...}]}
- a bare list: [{"scenario_id", "narrative", ...}]
- plain questions: [{"id", "text", ...}]

Only the narrative ever reaches a model. Every other field of an item is carried through in ``meta``.
"""
import hashlib
import json
from dataclasses import dataclass, field


class InputError(ValueError):
    pass


@dataclass
class Item:
    scenario_id: str
    narrative: str
    meta: dict = field(default_factory=dict)


def _items_of(doc):
    if isinstance(doc, dict) and isinstance(doc.get("scenarios"), list):
        return "team_c", doc["scenarios"]
    if isinstance(doc, list):
        if doc and isinstance(doc[0], dict) and "id" in doc[0] and "scenario_id" not in doc[0]:
            return "questions", doc
        return "list", doc
    raise InputError("input must be a Team C scenarios.json ({\"scenarios\": [...]}) or a JSON list of items")


def load(path, ids=None, limit=None):
    """Returns (items, info). ``ids``: keep only these scenario ids, in file order. ``limit``: first N."""
    raw = open(path, "rb").read()
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise InputError(f"{path}: not valid JSON: {e}")
    fmt, rows = _items_of(doc)
    id_key, text_key = ("id", "text") if fmt == "questions" else ("scenario_id", "narrative")
    items, seen, problems = [], set(), []
    for n, row in enumerate(rows):
        where = f"item {n}"
        if not isinstance(row, dict):
            problems.append(f"{where}: not an object")
            continue
        sid, text = row.get(id_key), row.get(text_key)
        if isinstance(sid, str) and sid:
            where = f"item {n} ({sid})"
        if not isinstance(sid, str) or not sid:
            problems.append(f"{where}: {id_key} must be a non-empty string")
        elif sid in seen:
            problems.append(f"{where}: duplicate {id_key}")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{where}: {text_key} must be a non-empty string")
        if problems and problems[-1].startswith(where):
            continue
        seen.add(sid)
        items.append(Item(sid, text, {k: v for k, v in row.items() if k not in (id_key, text_key)}))
    if problems:
        shown = "\n  ".join(problems[:20]) + (f"\n  ... {len(problems) - 20} more" if len(problems) > 20 else "")
        raise InputError(f"{path}: {len(problems)} problem(s):\n  {shown}")
    if ids is not None:
        wanted = list(ids)
        missing = [i for i in wanted if i not in seen]
        if missing:
            raise InputError(f"{len(missing)} requested id(s) not in {path}, e.g. {missing[0]}")
        keep = set(wanted)
        items = [it for it in items if it.scenario_id in keep]
    if limit is not None:
        items = items[:limit]
    info = {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "format": fmt,
            "count": len(items), "count_in_file": len(rows)}
    if fmt == "team_c":
        info["dataset"] = doc.get("dataset")
        info["codebook"] = doc.get("codebook")
    return items, info


def read_ids(path):
    """One scenario id per line; blank lines and # comments ignored."""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
