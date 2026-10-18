from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO / "tools" / "prepare_waist_crossing_muscle_surfaces.py"
spec = importlib.util.spec_from_file_location("waist_surface_tool", TOOL_PATH)
assert spec is not None and spec.loader is not None
waist_surface_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(waist_surface_tool)


def _vertex(x: float) -> bytes:
    return waist_surface_tool.VERTEX.pack(
        x, 0.0, 0.0, 0.0, 0.0, 1.0,
        0, 0, 0, 0,
        1.0, 0.0, 0.0, 0.0,
    )


def _payload(*, candidate: bool, corrupt_unsplit_vertex: bool = False) -> bytes:
    h = waist_surface_tool.HEADER
    r = waist_surface_tool.RECORD
    b = waist_surface_tool.BINDING
    if candidate:
        records = [
            (0, 1, 0, 4, 0, 3, 67, 1),
            (0, 1, 4, 3, 3, 3, 99, 1),
        ]
        vertices = b"".join(_vertex(float(i)) for i in (10, 11, 12, 13, 3, 4, 5))
        if corrupt_unsplit_vertex:
            changed = bytearray(vertices)
            changed[4 * waist_surface_tool.VERTEX.size] ^= 1
            vertices = bytes(changed)
        indices = np.asarray([0, 1, 2, 4, 5, 6], dtype="<u4").tobytes()
        header_counts = (7, 6)
    else:
        records = [
            (0, 1, 0, 3, 0, 3, 67, 1),
            (0, 1, 3, 3, 3, 3, 99, 1),
        ]
        vertices = b"".join(_vertex(float(i)) for i in range(6))
        indices = np.asarray([0, 1, 2, 3, 4, 5], dtype="<u4").tobytes()
        header_counts = (6, 6)
    header = h.pack(b"NHTISS4\0", 5, 2, 1, *header_counts,
                    0x12345678, b"s" * 32)
    binding = b.pack(7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    return (header + b"".join(r.pack(*row) for row in records) +
            binding + vertices + indices)


def test_preserves_unselected_surface_vertex_bytes_and_local_indices():
    result = waist_surface_tool.verify_source_preservation(
        _payload(candidate=False), _payload(candidate=True), {67}
    )
    assert result["byte_checked_unsplit_stable_ids"] == [99]
    assert result["unsplit_geometry_attribute_bytes_exact"]
    assert result["unsplit_local_triangle_indices_exact"]
    assert result["binding_table_bytes_exact"]


def test_rejects_an_unselected_surface_vertex_byte_change():
    with pytest.raises(ValueError, match="vertex bytes"):
        waist_surface_tool.verify_source_preservation(
            _payload(candidate=False),
            _payload(candidate=True, corrupt_unsplit_vertex=True),
            {67},
        )
