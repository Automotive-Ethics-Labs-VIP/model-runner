"""modelrunner: scenarios.json + a model -> output.json.

  modelrunner run --input scenarios.json --model deepseek-9b --out out/run.json     (on the login node: submits jobs)
  modelrunner run ... --engine mock --local                                         (anywhere, no GPU)
  modelrunner run ... --dry-run                                                     (show the plan and job scripts)
  modelrunner check-input scenarios.json                                            (validate and leak-check)
  modelrunner validate-output out/run.json
  modelrunner download --model org/name [--revision sha]                            (used by the download job)
"""
import argparse
import json
import os
import sys

from . import io_adapter, resolver, runner, slurm, writer
from .executor import Sampling


def add_run_args(p):
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True, help="registry key, org/name, or https://huggingface.co/org/name")
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--limit", type=int, help="run only the first N items")
    g.add_argument("--ids", help="file with one scenario id per line")
    p.add_argument("--engine", default="auto", choices=["auto", "vllm", "hf", "mock"])
    p.add_argument("--revision", help="pin a model commit (the login node pins one and passes it to the job)")
    p.add_argument("--template", default="team_c_decision_v1")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=3, help="1 + retries for replies that do not parse")
    p.add_argument("--resume", action="store_true", help="continue a run, skipping results already written")
    p.add_argument("--local", action="store_true", help="run here (inside a GPU allocation, or with --engine mock)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and job scripts; submit nothing")
    p.add_argument("--time", default="04:00:00", help="GPU job time limit")
    p.add_argument("--cpu-test", action="store_true",
                   help="submit to a CPU partition with the hf engine: tests the whole path without a GPU "
                        "(use a small model)")
    p.add_argument("--allow-remote-code", action="store_true",
                   help="allow trust_remote_code (only for registry entries marked trust_remote_code_ok)")


def _passthrough(args):
    """The run arguments to repeat inside the GPU job (absolute paths, pinned model)."""
    out = ["--input", os.path.abspath(args.input), "--out", os.path.abspath(args.out),
           "--model", args.model, "--engine", args.engine, "--template", args.template,
           "--n-samples", str(args.n_samples), "--temperature", str(args.temperature), "--top-p", str(args.top_p),
           "--max-new-tokens", str(args.max_new_tokens), "--seed", str(args.seed),
           "--max-attempts", str(args.max_attempts)]
    if args.limit is not None:
        out += ["--limit", str(args.limit)]
    if args.ids:
        out += ["--ids", os.path.abspath(args.ids)]
    if args.resume:
        out += ["--resume"]
    if args.allow_remote_code:
        out += ["--allow-remote-code"]
    return out


def cmd_run(args):
    engine = None if args.engine == "auto" else args.engine
    if args.cpu_test:
        if engine not in (None, "hf"):
            raise SystemExit("--cpu-test runs the hf engine (vLLM needs a GPU)")
        engine = args.engine = "hf"
    ids = io_adapter.read_ids(args.ids) if args.ids else None
    local = args.local or engine == "mock"
    offline = local and engine != "mock" and os.environ.get("HF_HUB_OFFLINE") == "1"
    model = resolver.resolve(args.model, engine=engine, allow_remote_code=args.allow_remote_code, offline=offline,
                             revision=args.revision)
    if offline and args.revision and model.revision != args.revision:
        raise SystemExit(f"{model.hf_id}@{args.revision} is not in the local cache")
    items, info = io_adapter.load(args.input, ids=ids, limit=args.limit)
    print(f"input: {info['count']} items from {args.input} ({info['format']})")
    print(f"model: {model.summary()}")
    if model.notes:
        print(f"note:  {model.notes}")
    if local:
        if args.dry_run:
            return 0
        if engine != "mock" and model.local_path is None and offline:
            raise SystemExit(f"{model.hf_id}@{model.revision} is not in the local cache; run the download job first")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        cfg = runner.RunConfig(input=args.input, out=args.out, model=model, template=args.template, ids=ids,
                               limit=args.limit, n_samples=args.n_samples, seed=args.seed,
                               max_attempts=args.max_attempts, resume=args.resume,
                               sampling=Sampling(args.temperature, args.top_p, args.max_new_tokens))
        runner.prepare(cfg)  # fail fast on leaks before loading anything
        runner.run(cfg)
        return 0
    # login node: pin the revision, download if needed, then the GPU job
    run_args = _passthrough(args)
    run_args[run_args.index("--model") + 1] = model.hf_id
    run_args += ["--revision", model.revision]
    jobs = []
    model.local_path = resolver.local_snapshot(model)
    if model.local_path is None:
        jobs.append(("download", slurm.download_job(model)))
    jobs.append(("run", slurm.run_job(model, os.path.abspath(args.out), run_args,
                                           time_limit=args.time, cpu_test=args.cpu_test)))
    if args.dry_run:
        for kind, job in jobs:
            print(f"\n===== {kind} job: {job.path}\n{job.script}")
        return 0
    after = None
    for kind, job in jobs:
        after = slurm.submit(job, after=after)
        print(f"{kind} job {after}: log {job.log.replace('%j', after)}")
    print(f"output: {os.path.abspath(args.out)} (partial results: {writer.results_path(os.path.abspath(args.out))})")
    print(f"status: squeue --me    details: sacct -j {after} --format=JobID,State,Elapsed,NodeList,ExitCode")
    return 0


def cmd_check_input(args):
    from . import prompt as prompt_mod
    items, info = io_adapter.load(args.input)
    tpl = prompt_mod.load_template(args.template)
    codes = prompt_mod.case_codes([it.scenario_id for it in items])
    leaky = [(it.scenario_id, prompt_mod.leaks(tpl.render(codes[it.scenario_id], it.narrative), it)) for it in items]
    leaky = [(s, l) for s, l in leaky if l]
    print(json.dumps({k: info[k] for k in ("format", "count", "sha256")}))
    for sid, found in leaky[:20]:
        print(f"LEAK {sid}: {', '.join(found)}")
    print("OK" if not leaky else f"{len(leaky)} prompt(s) would leak")
    return 1 if leaky else 0


def cmd_validate_output(args):
    with open(args.output, encoding="utf-8") as f:
        problems = writer.validate_output(json.load(f))
    for p in problems[:30]:
        print(p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_download(args):
    m = resolver.ResolvedModel(key=args.model, hf_id=args.model, revision=args.revision)
    print("downloaded to", resolver.download(m))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modelrunner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run"))
    p = sub.add_parser("check-input")
    p.add_argument("input")
    p.add_argument("--template", default="team_c_decision_v1")
    p = sub.add_parser("validate-output")
    p.add_argument("output")
    p = sub.add_parser("download")
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    args = ap.parse_args(argv)
    try:
        return {"run": cmd_run, "check-input": cmd_check_input, "validate-output": cmd_validate_output,
                "download": cmd_download}[args.cmd](args)
    except (io_adapter.InputError, resolver.ResolveError, runner.RunError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
