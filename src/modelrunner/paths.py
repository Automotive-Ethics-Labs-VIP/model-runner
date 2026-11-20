"""Where the repo's data files live (schemas, prompts, registry, slurm templates)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def repo(*parts):
    return os.path.join(ROOT, *parts)
