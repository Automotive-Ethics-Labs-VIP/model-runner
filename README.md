# model-runner

Run open-weight models over AEL scenario sets on SeaWulf. You give it a scenarios file and a model name; it
handles the rest (pinning the model, downloading weights, the GPU job, batching, parsing, retries) and writes
one `output.json`.

```
modelrunner run --input scenarios.json --model deepseek-9b --out runs/deepseek-round1.json
```

`--model` takes a [registry](registry/models.yaml) key, a Hugging Face id (`org/name`) or a URL
(`https://huggingface.co/org/name`). Models outside the registry are sized from the Hub's metadata.

## Status

| Phase | State |
| --- | --- |
| 1. Local pipeline (schemas, prompts, parser, mock engine, tests) | done: 50 tests, CI on Linux and Windows; a mock run over all 228 Round 1 scenarios validates |
| 0. Cluster checks | done: compute nodes reach huggingface.co and pypi.org; vLLM 0.34.0 installs; the 9B and Nanbeige 33B resolve (19.3 GB / 1 GPU, 66.2 GB / 2 GPUs) and both architectures are supported by vLLM without remote code; the full submit path (pin, download job, dependency, offline run with the hf engine, output) ran on a CPU node with `--cpu-test` |
| 0. GPU spike (vLLM, batched) | done (job 2443530, 2026-11-23): the 9B on one A100 80GB, vLLM 0.34.0; 18 Round 1 prompts in 66 s (582 tok/s) after an 11-minute load; the parser reads 15/18, the other 3 hit the 4096-token reply limit, so the defaults are now 8192 reply tokens and a 16384 context |
| 2. All of Round 1 on the 9B model | next |
| 3. Nanbeige 33B and models outside the registry | not started |

Inference only. Nothing here trains a model.

## Input

[`schemas/input.schema.json`](schemas/input.schema.json). One of:

- Team C's `scenarios.json` from the [scenario codebook](https://github.com/heinzarniaung/ael-codebook-v4-round1)
  (`python tools/build.py round1` writes `build/round1/expected/scenarios.json`)
- a list of `{"scenario_id", "narrative", ...}`
- a list of plain questions `{"id", "text"}`

Pick items with `--limit N` or `--ids ids.txt`. Check a file before running: `modelrunner check-input FILE`.

**The model sees only the prompt template and the narrative.** Each scenario gets an opaque case code
instead of its id, and every prompt is checked for leaks (ids, scene and variable names, Team C's
`S\d\d|anchor|twin|codebook` pattern) before anything runs. Labels and the record travel to the output's
`meta` for scoring; they are never sent.

## Output

[`schemas/output.schema.json`](schemas/output.schema.json): `run` (model and revision, engine and library
versions, hardware, Slurm job, sampling, prompt template and sha256, input sha256, timing, counts) and one
`results` entry per scenario and sample.

- `parsed` holds the reply's JSON; `parse_ok` says whether it is valid **for that scenario** (reply schema,
  case code, decision and ranking among the possible actions).
- A reply that does not parse is retried with a fresh sample (`--max-attempts`, default 3). Failed attempts
  are kept in `earlier_attempts`, and `run.counts.first_attempt_ok` reports the rate before retries.
- Replies cut off at `--max-new-tokens` are marked `finish_reason: "length"` and never count as answers.
- Join on `scenario_id`, never `case_code`.

While a run is going, each finished result is appended to `<out>.results.jsonl`; `--resume` continues a run
that was cut off.

## Prompt templates

[`prompts/team_c_decision_v1/`](prompts/team_c_decision_v1/) is Team C's `PROMPT_HEAD`, copied, with the reply
schema it asks for. Its sha256 is in every output. If Team C changes their prompt, add a new template
(`team_c_decision_v2`) rather than editing this one; a test flags the drift when their repo is checked out
next to this one.

## Running on SeaWulf

One-time, on the login node (`/gpfs/scratch/$USER/model-runner` holds envs, logs, jobs and the model cache):

```
git clone https://github.com/Automotive-Ethics-Labs-VIP/model-runner && cd model-runner
bash slurm/setup_env.sh vllm        # or: hf
```

Then, from that env (`/gpfs/scratch/$USER/model-runner/envs/vllm/bin/modelrunner`):

```
modelrunner run --input scenarios.json --model deepseek-9b --out runs/deepseek.json --dry-run   # look first
modelrunner run --input scenarios.json --model deepseek-9b --out runs/deepseek.json
```

On the login node this resolves the model to a commit, submits a CPU job to download it if it is not cached,
and submits the GPU job to start after it. The GPU job runs offline (`HF_HUB_OFFLINE=1`). Weights must be
safetensors; `trust_remote_code` is off unless a registry entry allows it. Models larger than one node
(4 x A100 80GB) are refused with the arithmetic.

Anywhere, without a GPU: `--engine mock --local` runs the whole pipeline with a deterministic fake model.
On the cluster, `--cpu-test` with a small model (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`) runs the real
download and run jobs on a CPU partition, to check the plumbing without waiting for a GPU.

## Layout

| Path | What |
| --- | --- |
| `src/modelrunner/` | `io_adapter` (input), `prompt` (templates, case codes, leak check), `resolver`, `executor/` (mock, vllm, hf), `parser`, `writer`, `runner`, `slurm`, `cli` |
| `schemas/` | input and output JSON Schemas |
| `prompts/` | versioned prompt templates with their reply schemas |
| `registry/models.yaml` | known models and their settings |
| `slurm/` | job templates and `setup_env.sh` |
| `spike/` | Phase 0 vLLM spike |
| `reference/` | the scripts proven by the first SeaWulf run (job 2417039) |
| `scripts/seawulf/` | how Claude Code runs commands over a shared ssh connection from Windows (WSL) |
| `examples/round1_anchors.json` | the 18 Round 1 anchor scenarios, for tests |
