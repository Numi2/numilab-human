from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_activation_sweep_audit import compile_audit


STDOUT = Path("Docs/media/native-activation-sweep-20260915/whole-body-support-wrench-stdout.txt")
STDERR = Path("Docs/media/native-activation-sweep-20260915/whole-body-support-wrench-stderr.txt")
PAYLOADS = {
    "rigid": {"path": "NumiHumanCurrent/myosim-fullbody-core-reference.nhrigid",
              "sha256": "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44"},
    "muscle": {"path": "NumiHumanCurrent/myosim-fullbody-muscle-reference.nhmyo",
               "sha256": "9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05"},
    "support": {"path": "NumiHumanCurrent/myosim-fullbody-support-contact.nhcnt",
                "sha256": "4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907"},
    "equalities": {"path": "NumiHumanCurrent/myosim-fullbody-joint-equalities.nheq",
                   "sha256": "b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a"},
    "tendon": {"path": "NumiHumanCurrent/numi-human-tendon-attachments.nhtendon",
               "sha256": "a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555"},
}


def test_activation_sweep_keeps_root_and_internal_results_separate() -> None:
    result = compile_audit(
        stdout=STDOUT,
        stderr=STDERR,
        source_commit="4581d7042ce4291715ff3893a9e1b171d80fa243",
        source_branch="detached",
        source_worktree_dirty=True,
        binary_sha256="67c0376a0c714650c17d58d1cf40c473bd18c817e1baa0caf5530a2609d9b5f5",
        payloads=PAYLOADS,
    )
    assert result["status"] == "partial"
    assert result["metrics"]["activation_sweeps"] == 32
    assert result["metrics"]["active_support_contacts"] == 9
    assert result["qualification"]["floating_root_wrench_closed"]
    assert result["qualification"]["body_weight_support_closed"]
    assert not result["qualification"]["internal_generalized_equilibrium"]
    assert result["ranked_internal_residuals"][0]["dof"] == 119
    assert result["ranked_internal_residuals"][0]["acceleration"] == pytest.approx(90.9280057267)
    assert not result["qualification"]["activation_calibration"]


def test_activation_sweep_rejects_non_bitwise_replay(tmp_path: Path) -> None:
    source = tmp_path / "stdout.txt"
    text = STDOUT.read_text(encoding="utf-8").replace("replay=bitwise", "replay=not_bitwise")
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ImportError, match="replay"):
        compile_audit(
            stdout=source,
            source_commit="fixture",
            source_branch="fixture",
            source_worktree_dirty=False,
            binary_sha256="0" * 64,
            payloads={},
        )


def test_activation_sweep_requires_ranked_internal_rows(tmp_path: Path) -> None:
    source = tmp_path / "stdout.txt"
    lines = [
        line for line in STDOUT.read_text(encoding="utf-8").splitlines()
        if not line.startswith("numi_human_whole_body_support_wrench=ok")
    ]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ImportError, match="whole-body support-wrench"):
        compile_audit(
            stdout=source,
            source_commit="fixture",
            source_branch="fixture",
            source_worktree_dirty=False,
            binary_sha256="0" * 64,
            payloads={},
        )
