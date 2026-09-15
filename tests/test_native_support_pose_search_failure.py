from __future__ import annotations

from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_support_pose_search_failure import compile_failure


STDOUT = Path("Docs/media/native-support-pose-search-20260915/native.stdout.txt")
STDERR = Path("Docs/media/native-support-pose-search-20260915/native.stderr.txt")


def test_support_pose_search_failure_is_retained_as_failure() -> None:
    result = compile_failure(
        stdout=STDOUT,
        stderr=STDERR,
        source_commit="663c82ac0e3f320a816e1b716a38b3a076924840",
        source_branch="human-completion-static-preload",
        source_worktree_dirty=False,
        binary_sha256="1d846358597401589af983f4bde3bbef53df6fd969421dcb512e2b035811380b",
        payloads={"support": {"path": "support.nhcnt", "sha256": "4d54f815" + "0" * 56}},
    )
    assert result["status"] == "failed"
    assert result["metrics"]["rejected_pose_candidates"] == 88
    assert result["metrics"]["separated_witnesses"] == 8
    assert not result["qualification"]["active_set_selected"]
    assert not result["qualification"]["anatomical_supports_loading"]


def test_support_pose_search_requires_fail_closed_error(tmp_path: Path) -> None:
    source = tmp_path / "stdout.txt"
    source.write_text(
        STDOUT.read_text(encoding="utf-8").replace(
            'myosim_articulated_visual=failed error="whole-body unilateral support wrench did not close"',
            'myosim_articulated_visual=ok',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ImportError, match="fail closed"):
        compile_failure(
            stdout=source,
            stderr=None,
            source_commit="fixture",
            source_branch="fixture",
            source_worktree_dirty=False,
            binary_sha256="0" * 64,
            payloads={},
        )
