import json
import os

import jsonschema
import pytest

from modelrunner import io_adapter, prompt
from modelrunner.paths import repo

SCHEMA = json.load(open(repo("schemas", "input.schema.json")))


def write(tmp_path, doc, name="in.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


def test_team_c_file(anchors_path, anchors):
    jsonschema.validate(anchors, SCHEMA)
    items, info = io_adapter.load(anchors_path)
    assert info["format"] == "team_c" and info["count"] == 18
    it = items[0]
    assert it.scenario_id == anchors["scenarios"][0]["scenario_id"]
    assert "labels" in it.meta and "narrative" not in it.meta and "scenario_id" not in it.meta


def test_list_and_questions(tmp_path):
    lst = [{"scenario_id": "a", "narrative": "x", "extra": 1}]
    qs = [{"id": "q1", "text": "Is it?"}]
    for doc, fmt in ((lst, "list"), (qs, "questions")):
        jsonschema.validate(doc, SCHEMA)
        items, info = io_adapter.load(write(tmp_path, doc))
        assert info["format"] == fmt and items[0].narrative in ("x", "Is it?")
    assert io_adapter.load(write(tmp_path, lst))[0][0].meta == {"extra": 1}


@pytest.mark.parametrize("doc, message", [
    ([{"scenario_id": "a", "narrative": ""}], "item 0 (a): narrative"),
    ([{"scenario_id": "a", "narrative": "x"}, {"scenario_id": "a", "narrative": "y"}], "item 1 (a): duplicate"),
    ([{"narrative": "x"}], "item 0: scenario_id"),
    ({"nope": []}, "must be a Team C"),
])
def test_errors_name_the_item_and_field(tmp_path, doc, message):
    with pytest.raises(io_adapter.InputError, match=message.replace("(", r"\(").replace(")", r"\)")):
        io_adapter.load(write(tmp_path, doc))


def test_ids_and_limit(anchors_path, anchors, tmp_path):
    ids = [s["scenario_id"] for s in anchors["scenarios"]][3:6]
    items, _ = io_adapter.load(anchors_path, ids=list(reversed(ids)))
    assert [i.scenario_id for i in items] == ids  # file order
    assert len(io_adapter.load(anchors_path, limit=4)[0]) == 4
    with pytest.raises(io_adapter.InputError, match="not in"):
        io_adapter.load(anchors_path, ids=["nope"])
    f = tmp_path / "ids.txt"
    f.write_text("# a comment\n" + "\n".join(ids) + "\n\n")
    assert io_adapter.read_ids(str(f)) == ids


def test_template_matches_team_c_packets():
    tpl = prompt.load_template("team_c_decision_v1")
    text = tpl.render("K7QX", "NARRATIVE HERE")
    assert text.startswith("You are the ethical decision head of an autonomous vehicle.")
    assert text.endswith("\nCASE K7QX:\nNARRATIVE HERE\n")
    assert len(tpl.sha256) == 64


def test_template_drift_against_team_c_checkout():
    """If a checkout of Team C's repo sits next to this one, their PROMPT_HEAD must still match ours."""
    import ast
    src = os.path.join(repo(), "..", "ael-codebook-v4-round1", "tools", "reference_pipeline.py")
    if not os.path.exists(src):
        pytest.skip("no ael-codebook-v4-round1 checkout next to this repo")
    tree = ast.parse(open(src, encoding="utf-8").read())
    head = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "PROMPT_HEAD" for t in n.targets))
    ours = prompt.load_template("team_c_decision_v1").text.split("\nCASE <<CASE_CODE>>:")[0]
    assert ours == head, "Team C changed PROMPT_HEAD: make a new template version instead of editing this one"


def test_case_codes_unique_stable_and_never_look_like_ids():
    ids = [f"scene_{i}-S{i % 30:02d}" for i in range(5000)]
    codes = prompt.case_codes(ids, seed=1)
    assert len(set(codes.values())) == len(ids)
    assert codes == prompt.case_codes(ids, seed=1)
    assert codes != prompt.case_codes(ids, seed=2)
    assert not any(prompt.LEAK_PATTERN.search(c) for c in codes.values())


def test_no_anchor_prompt_leaks(anchors_path):
    items, _ = io_adapter.load(anchors_path)
    tpl = prompt.load_template("team_c_decision_v1")
    codes = prompt.case_codes([i.scenario_id for i in items])
    for it in items:
        text = tpl.render(codes[it.scenario_id], it.narrative)
        assert prompt.leaks(text, it) == []
        # labels are action names, which the reply format lists anyway; structural names must never appear
        for value in (it.meta["situation"], it.meta["scene_id"], it.meta["anchor_id"], *it.meta["variables"]):
            if "_" in value:
                assert value not in text


def test_leak_check_catches_each_kind(anchors_path):
    it = io_adapter.load(anchors_path)[0][0]
    assert "scenario id" in prompt.leaks(f"... {it.scenario_id} ...", it)
    assert any("pattern" in x for x in prompt.leaks("this is the anchor case", it))
    assert any(it.meta["situation"] in x for x in prompt.leaks(f"situation {it.meta['situation']}", it))
