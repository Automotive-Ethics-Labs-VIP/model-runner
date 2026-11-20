"""vLLM offline batch inference: the whole batch goes to the engine at once."""
import time

from .base import Completion, Executor


class VllmExecutor(Executor):
    name = "vllm"

    def load(self, model):
        from vllm import LLM
        t0 = time.time()
        self.llm = LLM(model=model.local_path or model.hf_id, revision=model.revision, dtype=model.dtype,
                       tensor_parallel_size=model.gpus, max_model_len=model.max_model_len,
                       gpu_memory_utilization=0.9, trust_remote_code=model.trust_remote_code, seed=0)
        self.chat_kwargs = dict(model.chat_template_kwargs or {})
        return time.time() - t0

    def generate(self, requests, sampling):
        from vllm import SamplingParams
        params = [SamplingParams(temperature=sampling.temperature, top_p=sampling.top_p,
                                 max_tokens=sampling.max_new_tokens, seed=r.seed) for r in requests]
        messages = [[{"role": "user", "content": r.prompt}] for r in requests]
        kwargs = {"chat_template_kwargs": self.chat_kwargs} if self.chat_kwargs else {}
        outs = self.llm.chat(messages, params, use_tqdm=False, **kwargs)
        result = []
        for r, o in zip(requests, outs):
            c = o.outputs[0]
            result.append(Completion(r.key, c.text, "length" if c.finish_reason == "length" else "stop",
                                     len(o.prompt_token_ids), len(c.token_ids)))
        return result

    def info(self):
        import torch
        import transformers
        import vllm
        return {"name": "vllm", "version": vllm.__version__, "torch": torch.__version__,
                "transformers": transformers.__version__}

    def hardware(self):
        import torch
        return {"gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()}
