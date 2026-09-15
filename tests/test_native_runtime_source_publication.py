from __future__ import annotations

from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_runtime_source_publication import (
    compile_publication,
    immutable_write,
)


def test_public_source_tuple_contains_runtime_and_input_releases() -> None:
    result = compile_publication()

    assert result["runtime_source"]["immutable_tag"] == "human-native-step281-rank-audit-20260915"
    assert result["runtime_source"]["resolved_commit"] == "337741b51bfc4a5552a837dacb4d0b5c4268d298"
    assert result["source_input_package"]["immutable_tag"] == "human-native-runtime-source-inputs-20260915"
    assert set(result["inputs"]) == {
        "rigid", "muscle", "tendon", "support_contact", "joint_equalities",
    }
    assert result["inputs"]["rigid"]["bytes"] == 60324
    assert result["fresh_public_tag_replay"]["step_count"] == 64
    assert result["fresh_public_tag_replay"]["persistent_max_penetration_m"] == 0.0
    assert result["fresh_public_tag_replay"]["deterministic_replay"] == "bitwise"
    assert result["fresh_public_tag_replay"]["recruited_muscles"] == 416
    assert result["qualification"]["immutable_runtime_source"]
    assert result["qualification"]["published_runtime_input_package"]
    assert not result["qualification"]["temporal_force_convergence"]
    assert not result["qualification"]["historical_binary_reproduced"]


def test_public_source_tuple_refuses_an_alternate_evidence_root(tmp_path: Path) -> None:
    with pytest.raises(ImportError, match="native runtime source publication"):
        compile_publication(evidence=tmp_path)


def test_public_source_tuple_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_publication()
    output = tmp_path / "receipt.json"
    digest = immutable_write(output, result)
    assert immutable_write(output, compile_publication()) == digest
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(output, {**result, "status": "changed"})
