"""Run a model over the items on this machine (inside a GPU allocation, or anywhere with --engine mock).

input -> prompts (leak-checked) -> batched generation -> parse and validate -> retry failures with a
fresh sample -> results.jsonl (as it goes) -> output.json.

Every attempt's raw reply is kept: a retry adds a sample, it never silently replaces one, so the
first-attempt parse rate is reported next to the final one.
"""
import datetime
import hashlib
import os
import socket
import time
from dataclasses import asdict, dataclass

from . import OUTPUT_SCHEMA_VERSION, __version__, io_adapter, parser, prompt as prompt_mod, writer
from .executor import Request, Sampling, make


class RunError(RuntimeError):
    pass


@dataclass
class RunConfig:
    input: str
    out: str
    model: object                  # resolver.ResolvedModel
    template: str = "team_c_decision_v1"
    ids: list = None
    limit: int = None
    n_samples: int = 1
    seed: int = 0
    max_attempts: int = 3          # first try + 2 retries
    chunk: int = 256               # requests per generate() call; results are saved after each
    resume: bool = False
    sampling: Sampling = None


def request_seed(seed, scenario_id, sample_index, attempt):
    h = hashlib.sha256(f"{seed}:{scenario_id}:{sample_index}:{attempt}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def prepare(cfg):
    """Items, template, case codes and prompts. Refuses to run if any prompt leaks."""
    items, info = io_adapter.load(cfg.input, ids=cfg.ids, limit=cfg.limit)
    if not items:
        raise RunError("no items to run")
    tpl = prompt_mod.load_template(cfg.template)
    codes = prompt_mod.case_codes([it.scenario_id for it in items], cfg.seed)
    prompts, problems = {}, []
    for it in items:
        p = tpl.render(codes[it.scenario_id], it.narrative)
        found = prompt_mod.leaks(p, it)
        if found:
            problems.append(f"{it.scenario_id}: {', '.join(found)}")
        prompts[it.scenario_id] = p
    if problems:
        raise RunError("refusing to run: prompts would leak scenario structure to the model:\n  "
                       + "\n  ".join(problems[:10]))
    return items, info, tpl, codes, prompts


def run(cfg, executor=None, log=print):
    t_start = time.time()
    sampling = cfg.sampling or Sampling()
    items, info, tpl, codes, prompts = prepare(cfg)
    by_id = {it.scenario_id: it for it in items}
    feasible = {}
    for it in items:
        feasible[it.scenario_id] = (parser.feasible_actions(it.meta, tpl.checks)
                                    or parser.feasible_from_prompt(prompts[it.scenario_id]))
    results = writer.ResultsLog(writer.results_path(cfg.out))
    done = results.existing() if cfg.resume else {}
    if not cfg.resume and os.path.exists(results.path):
        raise RunError(f"{results.path} exists; pass --resume to continue it, or remove it")
    tasks = [(it.scenario_id, k) for it in items for k in range(cfg.n_samples) if (it.scenario_id, k) not in done]
    log(f"{len(items)} items x {cfg.n_samples} sample(s): {len(tasks)} to run, {len(done)} already done")

    executor = executor or make(cfg.model.engine)
    load_s = executor.load(cfg.model) if tasks else 0.0
    t_gen = time.time()
    history = {t: [] for t in tasks}       # every attempt per task
    pending = list(tasks)
    for attempt in range(1, cfg.max_attempts + 1):
        if not pending:
            break
        still = []
        for start in range(0, len(pending), cfg.chunk):
            chunk = pending[start:start + cfg.chunk]
            reqs = [Request(f"{sid}#{k}", prompts[sid], request_seed(cfg.seed, sid, k, attempt)) for sid, k in chunk]
            t0 = time.time()
            outs = executor.generate(reqs, sampling)
            share = (time.time() - t0) / max(1, len(chunk))
            for (sid, k), req, c in zip(chunk, reqs, outs):
                p = parser.parse(c.text, c.finish_reason, tpl.reply_schema, tpl.checks, codes[sid], feasible[sid])
                history[(sid, k)].append({
                    "attempt": attempt, "seed": req.seed, "raw_response": c.text, "finish_reason": c.finish_reason,
                    "thinking": p.thinking, "parsed": p.parsed, "parse_ok": p.ok, "parse_error": p.error,
                    "tokens": {"prompt": c.prompt_tokens, "completion": c.completion_tokens},
                    "latency_s": round(share, 3)})
                if p.ok or attempt == cfg.max_attempts:
                    results.append(final_row(by_id[sid], codes[sid], k, history[(sid, k)]))
                else:
                    still.append((sid, k))
            log(f"attempt {attempt}: {start + len(chunk)}/{len(pending)} done")
        pending = still
    gen_s = time.time() - t_gen

    rows = sorted(results.existing().values(), key=lambda r: (r["scenario_id"], r["sample_index"]))
    order = {it.scenario_id: n for n, it in enumerate(items)}
    rows = [r for r in rows if r["scenario_id"] in order]
    rows.sort(key=lambda r: (order[r["scenario_id"]], r["sample_index"]))
    run_meta = run_metadata(cfg, sampling, tpl, info, executor, rows, load_s, gen_s, time.time() - t_start)
    doc = writer.write_output(cfg.out, run_meta, rows)
    executor.close()
    c = doc["run"]["counts"]
    log(f"wrote {cfg.out}: {c['total']} results, {c['parse_ok']} parsed "
        f"({c['first_attempt_ok']} on the first attempt), {c['parse_failed']} failed, {c['truncated']} truncated")
    return doc


def final_row(item, code, k, attempts):
    last = attempts[-1]
    return {
        "scenario_id": item.scenario_id, "case_code": code, "sample_index": k,
        "raw_response": last["raw_response"], "thinking": last["thinking"], "parsed": last["parsed"],
        "parse_ok": last["parse_ok"], "parse_error": last["parse_error"], "finish_reason": last["finish_reason"],
        "attempts": len(attempts), "earlier_attempts": attempts[:-1],
        "tokens": last["tokens"], "latency_s": last["latency_s"], "meta": item.meta,
    }


def run_metadata(cfg, sampling, tpl, info, executor, rows, load_s, gen_s, total_s):
    m = cfg.model
    hardware = {"host": socket.gethostname()}
    hardware.update(executor.hardware() if rows else {})
    slurm = {k: os.environ[v] for k, v in (("job_id", "SLURM_JOB_ID"), ("partition", "SLURM_JOB_PARTITION"))
             if os.environ.get(v)}
    created = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    run_id = created.strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha256(
        f"{m.hf_id}{info['sha256']}{cfg.seed}".encode()).hexdigest()[:6]
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": created.isoformat().replace("+00:00", "Z"),
        "modelrunner_version": __version__,
        "model": {"key": m.key, "hf_id": m.hf_id, "revision": m.revision, "dtype": m.dtype, "gpus": m.gpus},
        "engine": executor.info(),
        "hardware": hardware,
        "slurm": slurm or None,
        "sampling": dict(asdict(sampling), seed=cfg.seed, n_samples=cfg.n_samples, max_attempts=cfg.max_attempts),
        "prompt": {"template": tpl.name, "sha256": tpl.sha256},
        "input": {k: info[k] for k in ("path", "sha256", "count", "format")},
        "timing": {"load_s": round(load_s, 1), "generate_s": round(gen_s, 1), "total_s": round(total_s, 1)},
        "counts": {
            "total": len(rows),
            "parse_ok": sum(r["parse_ok"] for r in rows),
            "parse_failed": sum(not r["parse_ok"] for r in rows),
            "first_attempt_ok": sum(r["parse_ok"] and r["attempts"] == 1 for r in rows),
            "truncated": sum(r["finish_reason"] == "length" for r in rows),
            "completion_tokens": sum(r["tokens"]["completion"] + sum(a["tokens"]["completion"]
                                                                     for a in r["earlier_attempts"]) for r in rows),
        },
    }
