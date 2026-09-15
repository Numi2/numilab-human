from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.vessel_mass_moment_transport_candidate import (
    MOMENTS,
    REGIONAL_EXCHANGE,
    _write_immutable,
    compile_candidate,
    simulate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_transport_preserves_mass_momentum_and_replays() -> None:
    compiled = compile_candidate(velocity_mps=[0.12, -0.04, 0.03])
    result = simulate(compiled, steps=512, reject_step=37)
    assert result["accepted_steps"] == 511
    assert result["rejected_steps"] == 1
    assert result["conservation"]["mass_conserved"]
    assert result["conservation"]["linear_momentum_conserved"]
    assert result["rollback"]["rejected_candidate_state_neutral"]
    assert result["replay"] == "bitwise"


def test_transport_binds_the_current_exact_clock() -> None:
    result = compile_candidate()
    assert result["clock"]["nanoseconds"] == 12_500
    assert result["source"]["vessel_count"] == 6
    assert result["qualification"]["first_second_moment_transport"]
    assert not result["qualification"]["anatomical_vessel_lumen"]
    assert not result["qualification"]["mechanical_blood_mass_owner"]


def test_transport_rejects_changed_regional_exchange(tmp_path: Path) -> None:
    value = json.loads(REGIONAL_EXCHANGE.read_text(encoding="utf-8"))
    value["results"]["timestep_nanoseconds"] = 25_000
    path = tmp_path / "regional.json"
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    with pytest.raises(HumanImportError, match="regional exchange exact-clock"):
        compile_candidate(regional_exchange=path)


def test_transport_receipt_is_immutable(tmp_path: Path) -> None:
    compiled = compile_candidate()
    receipt = simulate(compiled, steps=4, reject_step=2)
    output = tmp_path / "receipt.json"
    first = _write_immutable(output, receipt)
    assert first == _write_immutable(output, receipt)
    changed = dict(receipt)
    changed["status"] = "changed"
    with pytest.raises(HumanImportError, match="immutable"):
        _write_immutable(output, changed)


def test_source_receipt_paths_are_regular_files() -> None:
    assert MOMENTS.is_file()
    assert REGIONAL_EXCHANGE.is_file()
