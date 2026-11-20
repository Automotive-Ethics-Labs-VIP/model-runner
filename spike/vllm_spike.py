"""Phase 0 spike: can vLLM run a model offline on SeaWulf, batched, on Team C's prompt?

Reads Team C scenarios, builds prompts from prompts/team_c_decision_v1/prompt.txt, generates them all in
one vLLM batch, and writes one JSON line per scenario plus a summary line. Raw outputs are kept so the
parser can be built from real model text.

Env: MODEL_ID, INPUT, OUT, MAX_TOKENS (default 4096), TEMPERATURE (0.7), TP (tensor parallel, default 1).
"""
import json
import os
import socket
import time

import torch
import transformers
import vllm
from vllm import LLM, SamplingParams

MODEL_ID = os.environ["MODEL_ID"]
INPUT = os.environ["INPUT"]
OUT = os.environ["OUT"]
TEMPLATE = os.environ.get("TEMPLATE", "prompt.txt")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TP = int(os.environ.get("TP", "1"))


def main():
    scenarios = json.load(open(INPUT))["scenarios"]
    template = open(TEMPLATE, encoding="utf-8").read()
    prompts = [template.replace("<<CASE_CODE>>", f"T{i:03d}").replace("<<NARRATIVE>>", s["narrative"])
               for i, s in enumerate(scenarios)]
    messages = [[{"role": "user", "content": p}] for p in prompts]

    t0 = time.time()
    llm = LLM(model=MODEL_ID, dtype="bfloat16", tensor_parallel_size=TP, seed=0,
              max_model_len=8192, gpu_memory_utilization=0.9)
    load_s = time.time() - t0
    params = SamplingParams(temperature=TEMPERATURE, top_p=0.9, max_tokens=MAX_TOKENS, seed=0)

    t1 = time.time()
    outs = llm.chat(messages, params, use_tqdm=False)
    gen_s = time.time() - t1

    total_out = 0
    with open(OUT, "w") as f:
        for i, (s, o) in enumerate(zip(scenarios, outs)):
            c = o.outputs[0]
            total_out += len(c.token_ids)
            f.write(json.dumps({"i": i, "scenario_id": s["scenario_id"], "case_code": f"T{i:03d}",
                                "prompt_tokens": len(o.prompt_token_ids), "completion_tokens": len(c.token_ids),
                                "finish_reason": c.finish_reason, "text": c.text}) + "\n")
        f.write(json.dumps({"summary": {
            "model": MODEL_ID, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
            "gpus": torch.cuda.device_count(), "vllm": vllm.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "n": len(outs), "load_s": round(load_s, 1),
            "generate_s": round(gen_s, 1), "completion_tokens": total_out,
            "tokens_per_s": round(total_out / gen_s, 1),
            "finish_reasons": {r: sum(o.outputs[0].finish_reason == r for o in outs)
                               for r in set(o.outputs[0].finish_reason for o in outs)}}}) + "\n")
    print(f"done: {len(outs)} prompts, {total_out} tokens in {gen_s:.1f}s = {total_out / gen_s:.1f} tok/s; load {load_s:.1f}s")


if __name__ == "__main__":
    main()
