from __future__ import annotations

import struct

import pytest

from numilab_human.addbiomechanics_acquisition import (
    _activity,
    _coordinate_names,
    _frame,
    _parse_header,
    canonical,
)
from numilab_human.model import ImportError


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _field(number: int, wire: int, value: bytes | int) -> bytes:
    result = _varint((number << 3) | wire)
    if wire == 0:
        return result + _varint(int(value))
    if wire == 1 or wire == 5:
        return result + bytes(value)
    return result + _varint(len(value)) + bytes(value)


def _packed(values: list[float]) -> bytes:
    return b"".join(struct.pack("<d", value) for value in values)


def _synthetic_b3d() -> tuple[bytes, bytes]:
    model = (
        '<Model name="fixture"><CoordinateSet>'
        '<Coordinate name="pelvis_tilt"/><Coordinate name="hip_flexion_r"/>'
        '<Coordinate name="knee_angle_r"/><Coordinate name="ankle_angle_r"/>'
        "</CoordinateSet></Model>"
    ).encode()
    processing_pass = _field(1, 0, 1) + _field(2, 2, model)
    trial = (
        _field(1, 2, b"Gait_fixture")
        + _field(3, 0, 1)
        + _field(4, 1, struct.pack("<d", 0.01))
        + _field(5, 2, processing_pass)
    )
    header = (
        _field(1, 0, 4)
        + _field(3, 0, 1)
        + _field(4, 0, 0)  # replaced after the processing-pass frame is encoded
        + _field(5, 2, processing_pass)
        + _field(6, 2, b"calcn_r")
        + _field(9, 2, trial)
        + _field(10, 0, 4)
        + _field(11, 2, b"https://example.invalid/subject_1/")
        + _field(12, 2, b"fixture")
        + _field(13, 2, b"male")
        + _field(14, 1, struct.pack("<d", 1.78))
        + _field(15, 1, struct.pack("<d", 65.5))
        + _field(16, 0, 43)
        + _field(23, 2, b"healthy")
    )
    frame = (
        _field(1, 2, _packed([0.1, 0.2, 0.3, 0.4]))
        + _field(4, 2, _packed([1.0, 2.0, 3.0, 4.0]))
        + _field(8, 2, _packed([0.0, 100.0, 0.0]))
    )
    # Rebuild the header with the actual pass frame size, preserving the
    # source parser's length-delimited frame contract.
    header = header.replace(_field(4, 0, 0), _field(4, 0, len(frame)), 1)
    raw = struct.pack("<q", len(header)) + header + b"\x00" + frame
    return raw, frame


def test_source_header_and_frame_parser_preserve_measured_dimensions() -> None:
    raw, frame = _synthetic_b3d()
    header, offset = _parse_header(raw)
    assert header["sex"] == "male"
    assert header["age"] == 43
    assert header["coordinates"] == ["pelvis_tilt", "hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
    parsed, end = _frame(raw, offset, 1, len(frame), 1, 0, 4, 1)
    assert parsed == {
        "pos": [0.1, 0.2, 0.3, 0.4],
        "tau": [1.0, 2.0, 3.0, 4.0],
        "force": [0.0, 100.0, 0.0],
    }
    assert end == len(raw)


def test_header_rejects_duplicate_coordinate_names() -> None:
    with pytest.raises(ImportError, match="coordinate names are duplicated"):
        _coordinate_names('<Model><Coordinate name="knee"/><Coordinate name="knee"/></Model>')


def test_activity_classifier_keeps_stair_and_recovery_distinct() -> None:
    assert _activity("Gait_5_segment_0") == ("walking", "gait")
    assert _activity("StairUp_4_segment_0") == ("walking", "stair")
    assert _activity("Standing_01") == ("standing", "stand")
    assert _activity("Recovery_push") == ("recovery", "recovery")


def test_canonical_rejects_nonfinite_values() -> None:
    with pytest.raises(ImportError, match="non-finite"):
        canonical({"value": float("nan")})
