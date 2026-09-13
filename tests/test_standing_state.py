from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.standing_state import audit, decode_initial_state


def _state(*, abi: int = 2, activation: float = 0.2, velocity: float = 0.0) -> bytes:
    q = [0.0] * 129
    v = [0.0] * 128
    v[0] = velocity
    muscles = []
    for _ in range(416):
        muscles.extend((activation, activation, 0.10, 0.0))
    values = struct.pack("<1921f", *(q + v + muscles))
    source = bytes.fromhex("28" * 32)
    if abi == 1:
        header = struct.pack(
            "<8s8I3Q32s", b"NHINIT1\0", 1, 96, 129, 128, 416, 4, 0, 0,
            0x11, 0x22, 100, source,
        )
        return header + values
    header = bytearray(160)
    struct.pack_into(
        "<8s8I3Q32s", header, 0, b"NHINIT2\0", 2, 160, 129, 128, 416, 4, 1, 0,
        0x11, 0x22, 0, source,
    )
    struct.pack_into("<Q", header, 96, 12_500)
    return bytes(header) + values


def _args(path: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        initial_state=path,
        maximum_initial_speed=1.0e-6,
        maximum_fiber_speed=1.0e-6,
        activation_epsilon=1.0e-6,
        output=output,
    )


def test_decodes_both_prepared_state_abis() -> None:
    legacy = decode_initial_state(_state(abi=1))
    exact = decode_initial_state(_state(abi=2))
    assert legacy["abi"] == 1
    assert legacy["clock_nanoseconds"] == 100_000
    assert legacy["clock_authority"] == "legacy_microseconds"
    assert exact["abi"] == 2
    assert exact["clock_nanoseconds"] == 12_500
    assert exact["clock_authority"] == "nanoseconds"
    assert len(exact["q"]) == 129
    assert len(exact["v"]) == 128
    assert len(exact["muscles"]) == 416


def test_nonmaximal_stationary_recruitment_is_a_candidate(tmp_path: Path) -> None:
    initial = tmp_path / "prepared.nhinit"
    initial.write_bytes(_state(activation=0.2))
    output = tmp_path / "receipt.json"
    assert audit(_args(initial, output)) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "standing_state_candidate"
    assert receipt["qualification"]["standing_initial_state_candidate"]
    assert receipt["state"]["activation_nonzero_count"] == 416
    assert not receipt["state"]["uniform_maximal_activation"]
    assert not receipt["qualification"]["force_convergence"]


def test_uniform_maximal_activation_remains_diagnostic_only(tmp_path: Path) -> None:
    initial = tmp_path / "maximal.nhinit"
    initial.write_bytes(_state(activation=1.0))
    output = tmp_path / "receipt.json"
    assert audit(_args(initial, output)) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "partial"
    assert receipt["state"]["uniform_maximal_activation"]
    assert not receipt["qualification"]["standing_initial_state_candidate"]
    assert "force-path diagnostic" in receipt["gate"]["reasons"][0]


def test_moving_state_is_not_a_standing_candidate(tmp_path: Path) -> None:
    initial = tmp_path / "moving.nhinit"
    initial.write_bytes(_state(activation=0.2, velocity=2.0e-3))
    output = tmp_path / "receipt.json"
    assert audit(_args(initial, output)) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "partial"
    assert not receipt["qualification"]["stationary_generalized_state"]


def test_invalid_dimensions_fail_closed() -> None:
    raw = bytearray(_state())
    struct.pack_into("<I", raw, 16, 130)
    with pytest.raises(ImportError):
        decode_initial_state(bytes(raw))
