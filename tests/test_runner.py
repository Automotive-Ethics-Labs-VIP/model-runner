import json
import os

import pytest

from modelrunner import resolver, runner, writer
from modelrunner.cli import main
from modelrunner.executor import Completion, Executor
from modelrunner.executor.mock import MockExecutor


def cfg(tmp_path, anchors_path, model, **kw):
    return runner.RunConfig(input=anchors_path, out=str(tmp_path / "out.json"), model=model, **kw)


def test_mock_run_produces_valid_output(tmp_path, anchors_path, mock_model):
    doc = runner.run(cfg(tmp_path, anchors_path, mock_model), log=lambda *_: None)
    assert writer.validate_output(doc) == []
    c = doc["run"]["counts"]
    assert c["total"] == 18 and c["parse_ok"] + c["parse_failed"] == 18
    r = doc["results"][0]
    assert r["meta"]["labels"]  # carried through for scoring
    assert r["case_code"] in r["raw_response"]
    assert doc["run"]["prompt"]["template"] == "team_c_decision_v1"
    assert doc["run"]["input"]["count"] == 18


def test_same_seed_same_output_different_seed_differs(tmp_path, anchors_path, mock_model):
    a = runner.run(cfg(tmp_path / "a", anchors_path, mock_model), log=lambda *_: None)
    b = runner.run(cfg(tmp_path / "b", anchors_path, mock_model), log=lambda *_: None)
    c = runner.run(cfg(tmp_path / "c", anchors_path, mock_model, seed=7), log=lambda *_: None)
    key = lambda d: [(r["scenario_id"], r["raw_response"]) for r in d["results"]]
    assert key(a) == key(b)
    assert key(a) != key(c)


class Flaky(Executor):
    """Bad reply on attempt 1 for every prompt, good afterwards (via the mock)."""
    name = "mock"

    def __init__(self):
        self.mock, self.calls = MockExecutor(), 0

    def load(self, model):
        return 0.0

    def generate(self, requests, sampling):
        self.calls += 1
        if self.calls == 1:
            return [Completion(r.key, "</think>I refuse to answer in JSON.", "stop", 10, 7) for r in requests]
        outs = self.mock.generate(requests, sampling)
        for o in outs:  # make every retry well formed
            o.text, o.finish_reason = o.text, "stop"
        return outs


def test_retries_keep_every_attempt(tmp_path, anchors_path, mock_model):
    doc = runner.run(cfg(tmp_path, anchors_path, mock_model, limit=3), executor=Flaky(), log=lambda *_: None)
    for r in doc["results"]:
        assert r["attempts"] >= 2
        first = r["earlier_attempts"][0]
        assert first["raw_response"] == "</think>I refuse to answer in JSON." and first["parse_error"]
    assert doc["run"]["counts"]["first_attempt_ok"] == 0


class Dies(Executor):
    name = "mock"

    def __init__(self, after):
        self.mock, self.after, self.seen = MockExecutor(), after, 0

    def load(self, model):
        return 0.0

    def generate(self, requests, sampling):
        self.seen += len(requests)
        if self.seen > self.after:
            raise RuntimeError("job hit its time limit")
        return self.mock.generate(requests, sampling)


def test_resume_after_a_crash(tmp_path, anchors_path, mock_model):
    c = cfg(tmp_path, anchors_path, mock_model, chunk=5)
    with pytest.raises(RuntimeError):
        runner.run(c, executor=Dies(after=10), log=lambda *_: None)
    saved = writer.ResultsLog(writer.results_path(c.out)).existing()
    assert 0 < len(saved) <= 10
    with pytest.raises(runner.RunError, match="--resume"):
        runner.run(c, executor=MockExecutor(), log=lambda *_: None)
    c.resume = True
    counting = Dies(after=10 ** 6)
    doc = runner.run(c, executor=counting, log=lambda *_: None)
    assert doc["run"]["counts"]["total"] == 18
    assert len({r["scenario_id"] for r in doc["results"]}) == 18
    with open(writer.results_path(c.out), encoding="utf-8") as f:
        assert sum(1 for _ in f) == 18  # finished items were not run or written again
    reference = runner.run(cfg(tmp_path / "ref", anchors_path, mock_model), log=lambda *_: None)
    assert [r["raw_response"] for r in doc["results"]] == [r["raw_response"] for r in reference["results"]]


def test_n_samples(tmp_path, anchors_path, mock_model):
    doc = runner.run(cfg(tmp_path, anchors_path, mock_model, limit=2, n_samples=3), log=lambda *_: None)
    assert [(r["scenario_id"], r["sample_index"]) for r in doc["results"]][:3] == \
        [(doc["results"][0]["scenario_id"], k) for k in range(3)]
    assert len({r["raw_response"] for r in doc["results"][:3]}) > 1


def test_refuses_leaky_input(tmp_path, mock_model):
    path = tmp_path / "leaky.json"
    path.write_text(json.dumps([{"scenario_id": "x-S01", "narrative": "This is the anchor of the scene."}]))
    with pytest.raises(runner.RunError, match="leak"):
        runner.run(runner.RunConfig(input=str(path), out=str(tmp_path / "o.json"), model=mock_model),
                   log=lambda *_: None)


def test_cli_mock_run_and_validate(tmp_path, anchors_path, capsys):
    out = str(tmp_path / "cli.json")
    assert main(["run", "--input", anchors_path, "--model", "deepseek-9b", "--engine", "mock",
                 "--out", out, "--limit", "5"]) == 0
    assert main(["validate-output", out]) == 0
    assert main(["check-input", anchors_path]) == 0
    assert "OK" in capsys.readouterr().out


# ---- resolver --------------------------------------------------------------

class Sib:
    def __init__(self, name, size):
        self.rfilename, self.size = name, size


class FakeHub:
    def __init__(self, files, sha="abc123"):
        self.files, self.sha = files, sha

    def model_info(self, hf_id, revision=None, files_metadata=False):
        class Info:
            pass
        i = Info()
        i.sha, i.siblings = self.sha, [Sib(n, s) for n, s in self.files.items()]
        return i


def test_resolve_addresses():
    assert resolver.normalise("deepseek-9b")[0] == "deepseek-9b"
    assert resolver.normalise("https://huggingface.co/Nanbeige/Nanbeige4.3-33B")[1] == "Nanbeige/Nanbeige4.3-33B"
    assert resolver.normalise("https://huggingface.co/Nanbeige/Nanbeige4.3-33B/tree/main")[0] == "nanbeige-33b"
    assert resolver.normalise("someone/new-model") == (None, "someone/new-model")
    with pytest.raises(resolver.ResolveError):
        resolver.normalise("not a model")


@pytest.mark.parametrize("gb, gpus", [(19.3, 1), (66, 2), (100, 2), (150, 4), (200, 4)])
def test_gpu_sizing(gb, gpus):
    hub = FakeHub({"model.safetensors": int(gb * 1e9), "config.json": 1})
    assert resolver.resolve("someone/m", hub=hub).gpus == gpus


def test_too_big_shows_the_arithmetic():
    hub = FakeHub({"model.safetensors": int(1500e9)})
    with pytest.raises(resolver.ResolveError, match=r"1500 GB of weights x 1.2"):
        resolver.resolve("moonshot/huge", hub=hub)


def test_pins_revision_and_refuses_pickles_and_remote_code():
    m = resolver.resolve("someone/m", hub=FakeHub({"a.safetensors": 10, "config.json": 1}, sha="deadbeef"))
    assert m.revision == "deadbeef"
    with pytest.raises(resolver.ResolveError, match="pickle"):
        resolver.resolve("someone/m", hub=FakeHub({"pytorch_model.bin": 10}))
    with pytest.raises(resolver.ResolveError, match="trust_remote_code"):
        resolver.resolve("someone/m", allow_remote_code=True, hub=FakeHub({"a.safetensors": 1}))


def test_dry_run_renders_both_jobs(tmp_path, anchors_path, monkeypatch, capsys):
    monkeypatch.setenv("MODELRUNNER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(resolver, "resolve", lambda *a, **k: resolver.ResolvedModel(
        key="x", hf_id="someone/m", revision="deadbeef", gpus=2, weights_bytes=int(66e9)))
    assert main(["run", "--input", anchors_path, "--model", "someone/m", "--out", str(tmp_path / "o.json"),
                 "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "download job" in out and "run job" in out
    assert "--gres=gpu:2" in out and "-m modelrunner run --local" in out
    assert "download --model someone/m --revision deadbeef" in out
    assert "HF_HUB_OFFLINE=1" in out
    assert "--revision deadbeef" in out.split("===== run job")[1]  # the job uses the pinned commit
    assert not os.path.exists(tmp_path / "home" / "jobs")  # nothing written or submitted


def test_cpu_test_job_has_no_gpu(tmp_path, anchors_path, monkeypatch, capsys):
    monkeypatch.setenv("MODELRUNNER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(resolver, "resolve", lambda *a, **k: resolver.ResolvedModel(
        key="x", hf_id="someone/tiny", revision="abc", engine=k.get("engine") or "vllm"))
    assert main(["run", "--input", anchors_path, "--model", "someone/tiny", "--out", str(tmp_path / "o.json"),
                 "--cpu-test", "--dry-run"]) == 0
    run_script = capsys.readouterr().out.split("===== run job")[1]
    assert "--gres" not in run_script and "short-40core-shared" in run_script
    assert "--engine hf" in run_script
