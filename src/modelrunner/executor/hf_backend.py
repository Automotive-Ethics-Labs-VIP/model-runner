"""transformers fallback: batched generate with left padding, prompts sorted by length.

This is the stack proven on SeaWulf (job 2417039), made batched. Slower than vLLM but has fewer moving
parts. Per-request seeds are not supported by batched generate; the batch uses the first request's seed.
"""
import time

from .base import Completion, Executor

LOADERS = ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM")


class HfExecutor(Executor):
    name = "hf"
    batch_size = 16

    def load(self, model):
        import torch
        import transformers
        t0 = time.time()
        src = model.local_path or model.hf_id
        self.tok = transformers.AutoTokenizer.from_pretrained(src, revision=model.revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        names = [model.loader_class] if model.loader_class else list(LOADERS)
        errors = []
        for name in names:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(src, revision=model.revision, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=model.trust_remote_code)
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}: {e}")
        else:
            raise RuntimeError("no loader class could load the model: " + " | ".join(errors))
        self.model.eval()
        self.chat_kwargs = dict(model.chat_template_kwargs or {})
        return time.time() - t0

    def generate(self, requests, sampling):
        import torch
        texts = [self.tok.apply_chat_template([{"role": "user", "content": r.prompt}], tokenize=False,
                                              add_generation_prompt=True, **self.chat_kwargs) for r in requests]
        order = sorted(range(len(requests)), key=lambda i: len(texts[i]))
        done = {}
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            torch.manual_seed(requests[idx[0]].seed)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=sampling.max_new_tokens, do_sample=True,
                                          temperature=sampling.temperature, top_p=sampling.top_p,
                                          pad_token_id=self.tok.pad_token_id)
            n_in = enc["input_ids"].shape[1]
            for row, i in enumerate(idx):
                new = out[row][n_in:]
                n_prompt = int(enc["attention_mask"][row].sum())
                ended = (new == self.tok.eos_token_id).nonzero()
                n_new = int(ended[0]) + 1 if len(ended) else len(new)
                finish = "stop" if len(ended) or n_new < sampling.max_new_tokens else "length"
                text = self.tok.decode(new[:n_new], skip_special_tokens=True)
                done[i] = Completion(requests[i].key, text, finish, n_prompt, n_new)
        return [done[i] for i in range(len(requests))]

    def info(self):
        import torch
        import transformers
        return {"name": "hf", "version": transformers.__version__, "torch": torch.__version__,
                "transformers": transformers.__version__}

    def hardware(self):
        import torch
        return {"gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()}
