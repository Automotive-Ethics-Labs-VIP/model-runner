#!/usr/bin/env python3
"""Run a few prompts through a Hugging Face model and save the responses to a .txt (proof of run).

Env vars: MODEL_ID, MAX_NEW_TOKENS (default 1536), OUT_PATH.
"""
import datetime
import os
import socket
import time

import torch
import transformers
from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "1536"))
JOB = os.environ.get("SLURM_JOB_ID", "local")
OUT = os.environ.get("OUT_PATH", f"responses-{JOB}.txt")

PROMPTS = [
    "In two sentences, what is a Slurm job array and when would you use one?",
    "A train leaves a station at 3:15 PM going 60 mph. A second train leaves the same station at "
    "4:00 PM going 80 mph on the same track. At what time does the second train catch up? Show your reasoning.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
]


def load_model(model_id):
    # The HF page loads this model as multimodal (AutoModelForMultimodalLM); fall back if the
    # installed transformers does not have that class or it cannot load the checkpoint.
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            print(f"{name}: not in this transformers version", flush=True)
            continue
        try:
            model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
            print(f"loaded with {name}", flush=True)
            return model, name
        except Exception as e:  # noqa: BLE001
            print(f"{name} failed: {type(e).__name__}: {e}", flush=True)
    raise SystemExit("No auto class could load the model; see messages above.")


def main():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"
    print(f"model={MODEL_ID} gpu={gpu} host={socket.gethostname()}", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; refusing to run on CPU.")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model, cls_name = load_model(MODEL_ID)
    model.eval()
    load_s = time.time() - t0

    with open(OUT, "w") as f:
        f.write("=== RUN PROOF ===\n")
        f.write(f"date:         {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"slurm job id: {JOB}\n")
        f.write(f"host:         {socket.gethostname()}\n")
        f.write(f"gpu:          {gpu}\n")
        f.write(f"model:        {MODEL_ID}\n")
        f.write(f"loader class: {cls_name}\n")
        f.write(f"torch {torch.__version__} | transformers {transformers.__version__}\n")
        f.write(f"load time:    {load_s:.1f}s | max_new_tokens: {MAX_NEW}\n")
        f.write(f"gpu mem after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n\n")
        f.flush()

        for i, prompt in enumerate(PROMPTS, 1):
            messages = [{"role": "user", "content": prompt}]
            inputs = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(model.device)
            n_in = inputs["input_ids"].shape[1]
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                                     temperature=0.7, top_p=0.9)
            dt = time.time() - t1
            new_tokens = out[0][n_in:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            f.write(f"--- PROMPT {i} ---\n{prompt}\n\n--- RESPONSE {i} "
                    f"({len(new_tokens)} tokens, {dt:.1f}s, {len(new_tokens) / dt:.1f} tok/s) ---\n{text}\n\n")
            f.flush()
            print(f"prompt {i} done: {len(new_tokens)} tokens in {dt:.1f}s", flush=True)
            if len(new_tokens) >= MAX_NEW:
                print(f"  note: prompt {i} hit max_new_tokens; the reasoning model may still be mid-thought", flush=True)

    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
