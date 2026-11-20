import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

ANCHORS = os.path.join(ROOT, "examples", "round1_anchors.json")


@pytest.fixture
def anchors_path():
    return ANCHORS


@pytest.fixture
def anchors():
    with open(ANCHORS, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_model():
    from modelrunner.resolver import ResolvedModel
    return ResolvedModel(key="mock", hf_id="test/mock", engine="mock")
