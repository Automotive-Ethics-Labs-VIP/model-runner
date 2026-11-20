"""The engine interface: a batch of chat prompts in, one completion per prompt out."""
from dataclasses import dataclass


@dataclass
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 4096


@dataclass
class Request:
    key: str          # unique within a batch
    prompt: str       # sent as the single user message
    seed: int


@dataclass
class Completion:
    key: str
    text: str
    finish_reason: str          # "stop" or "length"
    prompt_tokens: int
    completion_tokens: int


class Executor:
    name = "base"

    def load(self, model):
        """Load weights. ``model``: a resolver.ResolvedModel. Returns seconds taken."""
        raise NotImplementedError

    def generate(self, requests, sampling):
        """One Completion per Request, in the same order."""
        raise NotImplementedError

    def info(self):
        """Engine name and library versions, for the output's run.engine."""
        return {"name": self.name}

    def hardware(self):
        return {}

    def close(self):
        pass


def make(name):
    if name == "mock":
        from .mock import MockExecutor
        return MockExecutor()
    if name == "vllm":
        from .vllm_backend import VllmExecutor
        return VllmExecutor()
    if name == "hf":
        from .hf_backend import HfExecutor
        return HfExecutor()
    raise ValueError(f"unknown engine {name!r} (mock, vllm, hf)")
