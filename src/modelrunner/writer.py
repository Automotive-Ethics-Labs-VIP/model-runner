"""Results on disk: an append-only results.jsonl while running, then one validated output.json.

Each finished (scenario, sample) is appended and flushed straight away, so a timeout loses nothing and
``--resume`` can skip what is already there.
"""
import json
import os

import jsonschema

from .paths import repo


def results_path(out_path):
    stem = out_path[:-5] if out_path.endswith(".json") else out_path
    return stem + ".results.jsonl"


class ResultsLog(object):
    def __init__(self, path):
        self.path = path

    def existing(self):
        """{(scenario_id, sample_index): row} already written; a torn last line is ignored."""
        rows = {}
        if not os.path.exists(self.path):
            return rows
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rows[(row["scenario_id"], row["sample_index"])] = row
        return rows

    def append(self, row):
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def output_schema():
    with open(repo("schemas", "output.schema.json"), encoding="utf-8") as f:
        return json.load(f)


def validate_output(doc):
    errors = sorted(jsonschema.Draft202012Validator(output_schema()).iter_errors(doc),
                    key=lambda e: list(map(str, e.absolute_path)))
    return [f"{'/'.join(map(str, e.absolute_path)) or 'output'}: {e.message}" for e in errors]


def write_output(path, run, results):
    doc = {"schema_version": run.pop("schema_version"), "run": run, "results": results}
    problems = validate_output(doc)
    if problems:
        raise ValueError("output does not match schemas/output.schema.json:\n  " + "\n  ".join(problems[:10]))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return doc
