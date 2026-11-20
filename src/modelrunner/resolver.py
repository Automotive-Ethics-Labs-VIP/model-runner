"""Model name -> a pinned, sized, safe-to-load model.

- Accepts a registry key, ``org/name``, or a ``https://huggingface.co/org/name`` URL.
- Pins the revision (commit sha) and records it in every output.
- Requires safetensors weights; refuses pickle-only repos.
- ``trust_remote_code`` stays off unless the registry entry allows it and the user asks.
- Sizes the job: gpus = smallest n with weights x 1.2 <= n x 80 GB x 0.9; v1 stops at one node (4 GPUs).
"""
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

from .paths import repo

GPU_BYTES = 80e9          # A100 80GB
GPU_USABLE = 0.9
OVERHEAD = 1.2            # KV cache and activations, rough
MAX_GPUS = 4              # one SeaWulf A100 node
_URL = re.compile(r"^https?://huggingface\.co/([^/]+/[^/?#]+)")
_HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
WEIGHT_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]


class ResolveError(ValueError):
    pass


@dataclass
class ResolvedModel:
    key: str
    hf_id: str
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    engine: str = "vllm"
    gpus: int = 1
    max_model_len: int = 8192
    weights_bytes: Optional[int] = None
    chat_template_kwargs: dict = field(default_factory=dict)
    loader_class: Optional[str] = None
    trust_remote_code: bool = False
    local_path: Optional[str] = None
    notes: str = ""

    def summary(self):
        size = f"{self.weights_bytes / 1e9:.1f} GB" if self.weights_bytes else "size unknown"
        return (f"{self.hf_id} @ {self.revision or 'unpinned'} ({size}, {self.dtype}) on {self.gpus} GPU(s), "
                f"engine {self.engine}")


def registry():
    with open(repo("registry", "models.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalise(name):
    """(registry key or None, hf_id)."""
    reg = registry()
    if name in reg:
        return name, reg[name]["hf_id"]
    m = _URL.match(name)
    hf_id = m.group(1) if m else name.strip().rstrip("/")
    if not _HF_ID.match(hf_id):
        raise ResolveError(f"{name!r} is neither a registry key ({', '.join(sorted(reg))}) nor a "
                           "Hugging Face address (org/name or https://huggingface.co/org/name)")
    for key, entry in reg.items():
        if entry["hf_id"] == hf_id:
            return key, hf_id
    return None, hf_id


def gpus_needed(weights_bytes):
    need = math.ceil(weights_bytes * OVERHEAD / (GPU_BYTES * GPU_USABLE))
    return max(1, need)


def size_check(weights_bytes, min_gpus=None):
    gpus = max(gpus_needed(weights_bytes), min_gpus or 1)
    if gpus > MAX_GPUS:
        raise ResolveError(
            f"too big for one node: {weights_bytes / 1e9:.0f} GB of weights x {OVERHEAD} overhead = "
            f"{weights_bytes * OVERHEAD / 1e9:.0f} GB, but one node has {MAX_GPUS} x {GPU_BYTES / 1e9:.0f} GB x "
            f"{GPU_USABLE} usable = {MAX_GPUS * GPU_BYTES * GPU_USABLE / 1e9:.0f} GB. Multi-node runs are not supported.")
    # tensor parallelism needs a GPU count that divides the attention heads; powers of two are safe
    return 1 << (gpus - 1).bit_length()


def check_files(filenames):
    names = list(filenames)
    has_safetensors = any(n.endswith(".safetensors") for n in names)
    pickles = [n for n in names if n.endswith((".bin", ".pt", ".pth", ".pkl", ".ckpt"))]
    if not has_safetensors:
        raise ResolveError("no .safetensors weights in the repo" +
                           (f" (only pickle-based files such as {pickles[0]}, which can run code when loaded)"
                            if pickles else ""))


def resolve(name, engine=None, allow_remote_code=False, offline=False, hub=None):
    """``hub``: a huggingface_hub.HfApi-like object (injectable for tests). ``offline``: use only the
    registry and the local cache."""
    key, hf_id = normalise(name)
    entry = registry().get(key, {}) if key else {}
    m = ResolvedModel(key=key or hf_id, hf_id=hf_id, revision=entry.get("revision"),
                      dtype=entry.get("dtype", "bfloat16"), engine=engine or entry.get("engine", "vllm"),
                      max_model_len=entry.get("max_model_len", 8192),
                      chat_template_kwargs=entry.get("chat_template_kwargs") or {},
                      loader_class=entry.get("loader_class"), notes=entry.get("notes", "").strip())
    if allow_remote_code:
        if not entry.get("trust_remote_code_ok"):
            raise ResolveError(f"{hf_id}: trust_remote_code is only allowed for registry entries marked "
                               "trust_remote_code_ok: it runs the repo's own Python with your cluster account")
        m.trust_remote_code = True
    if m.engine == "mock":
        return m
    if offline:
        m.local_path = local_snapshot(m)
        if m.local_path:
            files = [os.path.join(dp, f) for dp, _, fs in os.walk(m.local_path) for f in fs]
            check_files(files)
            m.weights_bytes = sum(os.path.getsize(f) for f in files if f.endswith(".safetensors"))
            m.gpus = size_check(m.weights_bytes, entry.get("min_gpus"))
        return m
    if hub is None:
        from huggingface_hub import HfApi
        hub = HfApi()
    try:
        info = hub.model_info(hf_id, revision=m.revision, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            raise ResolveError(f"{hf_id}: access denied. If the model is gated, accept its licence on "
                               "huggingface.co and set HF_TOKEN in the environment.")
        raise ResolveError(f"{hf_id}: could not read model info from the Hub: {type(e).__name__}: {msg[:300]}")
    files = {s.rfilename: (s.size or 0) for s in info.siblings}
    check_files(files)
    m.revision = info.sha
    m.weights_bytes = sum(size for n, size in files.items() if n.endswith(".safetensors"))
    m.gpus = size_check(m.weights_bytes, entry.get("min_gpus"))
    return m


def cache_dir():
    from huggingface_hub.constants import HF_HUB_CACHE
    return HF_HUB_CACHE


def local_snapshot(m):
    """Path of a downloaded snapshot of m (at m.revision, or the only/newest one), or None."""
    folder = os.path.join(cache_dir(), "models--" + m.hf_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(folder):
        return None
    if m.revision and os.path.isdir(os.path.join(folder, m.revision)):
        return os.path.join(folder, m.revision)
    snaps = sorted((os.path.getmtime(os.path.join(folder, s)), s) for s in os.listdir(folder))
    if not snaps:
        return None
    if m.revision is None:
        m.revision = snaps[-1][1]
        return os.path.join(folder, snaps[-1][1])
    return None


def download(m):
    """Fetch weights (safetensors only) at the pinned revision; returns the local path."""
    from huggingface_hub import snapshot_download
    m.local_path = snapshot_download(m.hf_id, revision=m.revision, allow_patterns=WEIGHT_PATTERNS)
    return m.local_path
