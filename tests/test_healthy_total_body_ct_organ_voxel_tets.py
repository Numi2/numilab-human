from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from numilab_human import healthy_total_body_ct_organ_voxel_tets as organs
from numilab_human.model import ImportError


def _nifti(values: list[float], shape: tuple[int, int, int]) -> bytes:
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, -1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    header[123] = 2
    struct.pack_into("<h", header, 252, 1)
    struct.pack_into("<h", header, 254, 0)
    struct.pack_into("<3f", header, 268, 10.0, 20.0, 30.0)
    header[344:348] = b"n+1\x00"
    payload = struct.pack("<" + "f" * len(values), *values)
    return gzip.compress(bytes(header) + payload, mtime=0)


def test_selected_label_indices_are_read_from_hash_bound_source_nifti(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _nifti([0, 4, 9, 4, 2, 9, 0, 2], (4, 2, 1))
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("scan-001.nii.gz", raw)
    monkeypatch.setattr(organs.source, "NIFTI_CHUNK_BYTES", 8)

    chunks, dimensions = organs._read_source_label_indices(
        archive_path=archive_path,
        member_name="scan-001.nii.gz",
        expected_nifti_sha256=hashlib.sha256(gzip.decompress(raw)).hexdigest(),
        expected_counts={4: 2, 9: 2},
        allowed_source_label_ids={2, 4, 9},
    )

    assert dimensions == [4, 2, 1]
    assert {label_id: sum(map(len, rows)) for label_id, rows in chunks.items()} == {4: 2, 9: 2}
    voxels_4 = organs._linear_indices_to_voxels(np.concatenate(chunks[4]), dimensions)
    voxels_9 = organs._linear_indices_to_voxels(np.concatenate(chunks[9]), dimensions)
    assert voxels_4 == {(1, 0, 0), (3, 0, 0)}
    assert voxels_9 == {(2, 0, 0), (1, 1, 0)}


def test_source_nifti_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    raw = _nifti([0, 4, 9, 4, 2, 9, 0, 2], (4, 2, 1))
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("scan-001.nii.gz", raw)

    with pytest.raises(ImportError, match="uncompressed source NIfTI hash"):
        organs._read_source_label_indices(
            archive_path=archive_path,
            member_name="scan-001.nii.gz",
            expected_nifti_sha256="0" * 64,
            expected_counts={4: 2, 9: 2},
            allowed_source_label_ids={2, 4, 9},
        )


def test_cli_summary_uses_current_surface_boundary_gate(monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    result = {
        "receipt_sha256": "a" * 64,
        "scan_id": "001",
        "meshes": [{"source_voxel_count": 2, "tetrahedron_count": 12}],
        "qualification": {
            "all_selected_registered_surfaces_match_source_mask_boundaries": True,
            "mechanics_admitted": False,
        },
    }
    monkeypatch.setattr(organs, "compile_candidates", lambda **_: result)

    assert organs.run(SimpleNamespace(plan=Path("unused"), output=Path("unused"))) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["all_source_boundaries_exact"] is True
    assert summary["mechanics_admitted"] is False
