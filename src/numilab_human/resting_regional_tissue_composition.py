"""Merge disjoint passive muscle repairs in the existing NHTISS ABI.

This composes already prepared source rows. It does not establish anatomical
clearance, accepted-motion validity, or native admission.
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .passive_attachment_composition import _read_nhtiss4, _row_slices, require


def merge_fixed_allocation_tissue_rows(
    base_payload: Path,
    replacements: Sequence[tuple[Path, Sequence[int]]],
    *,
    expected_sha256: Mapping[str, str],
) -> tuple[bytes, dict]:
    """Merge reviewed rows while retaining the body, material, and GID allocation."""
    def read_pinned(path):
        data = _read_nhtiss4(path)
        require(expected_sha256.get(str(data["path"])) == data["sha256"],
                "reviewed tissue input SHA256 changed or is missing")
        return data

    base = read_pinned(base_payload)
    base_rows = {int(r[6]): _row_slices(base, r) for r in base["records"]}
    selected: dict[int, dict] = {}
    inputs = []
    for path, stable_ids in replacements:
        ids = list(stable_ids)
        require(ids and all(type(sid) is int and sid in base_rows for sid in ids),
                "replacement IDs must be existing anatomical integers")
        require(len(set(ids)) == len(ids) and not set(ids).intersection(selected),
                "regional replacement rows overlap or repeat")
        candidate = read_pinned(path)
        require(candidate["fingerprint"] == base["fingerprint"] and
                candidate["source_digest"] == base["source_digest"],
                "registration or source archive identity changed")
        require(candidate["surface_count"] == base["surface_count"] and
                np.array_equal(candidate["records"][:, 6:], base["records"][:, 6:]),
                "ordered anatomical identities or layers changed")
        require(candidate["binding_count"] == base["binding_count"] and
                candidate["raw"][candidate["binding_start"]:candidate["vertex_start"]] ==
                base["raw"][base["binding_start"]:base["vertex_start"]],
                "attachment body binding changed")
        rows = {int(r[6]): _row_slices(candidate, r) for r in candidate["records"]}
        changed = set()
        for sid, old in base_rows.items():
            row = rows[sid]
            require(row["vertex_count"] == old["vertex_count"] and
                    row["first_vertex"] == old["first_vertex"],
                    "source vertex allocation changed")
            require(row["binding_bytes"] == old["binding_bytes"] and
                    row["first_binding"] == old["first_binding"] and
                    row["binding_count"] == old["binding_count"],
                    "attachment body binding changed")
            old_records = np.frombuffer(old["vertex_bytes"], dtype=np.uint8).reshape(-1, 56)
            new_records = np.frombuffer(row["vertex_bytes"], dtype=np.uint8).reshape(-1, 56)
            require(np.array_equal(old_records[:, 24:], new_records[:, 24:]),
                    "material parent lanes or weights changed")
            geometry = np.ndarray((row["vertex_count"], 6), dtype="<f4",
                                  buffer=row["vertex_bytes"], strides=(56, 4))
            require(np.isfinite(geometry).all(), "nonfinite source geometry")
            if row["vertex_bytes"] != old["vertex_bytes"] or row["local_faces"] != old["local_faces"]:
                changed.add(sid)
        require(changed == set(ids), "candidate changed an undeclared row or omitted a declared repair")
        for sid in ids:
            selected[sid] = rows[sid]
        inputs.append({"path": str(candidate["path"]), "sha256": candidate["sha256"],
                       "changed_stable_ids": ids})
    require(selected, "no regional repairs supplied")

    records = base["records"].copy()
    vertices, faces = [], []
    vertex_count = index_count = 0
    for index, record in enumerate(records):
        sid = int(record[6])
        row = selected.get(sid, base_rows[sid])
        local_faces = np.asarray(row["local_faces"], dtype="<u4")
        records[index, 2:6] = (vertex_count, row["vertex_count"], index_count, local_faces.size)
        vertices.append(row["vertex_bytes"])
        faces.append((local_faces + vertex_count).astype("<u4").tobytes())
        vertex_count += row["vertex_count"]
        index_count += local_faces.size
    require(vertex_count == base["vertex_count"], "combined GID allocation changed")
    header = bytearray(base["raw"][:base["vertex_start"]])
    header[64:64 + base["surface_count"] * 32] = records.astype("<u4").tobytes()
    struct.pack_into("<II", header, 20, vertex_count, index_count)
    raw = bytes(header) + b"".join(vertices) + b"".join(faces)
    require(len(raw) == 64 + base["surface_count"] * 32 + base["binding_count"] * 36 +
            vertex_count * 56 + index_count * 4, "combined tissue byte range mismatch")
    proof = {"base_path": str(base["path"]), "base_sha256": base["sha256"],
             "regional_inputs": inputs, "changed_stable_ids": sorted(selected),
             "output_sha256": hashlib.sha256(raw).hexdigest(),
             "source_vertex_allocation_unchanged": True,
             "body_bindings_and_material_weight_bytes_unchanged": True,
             "unrelated_row_geometry_unchanged": True,
             "scope": "Source row composition only; regional manifests and ancestry must be retained by the caller. "
                      "Not anatomical, accepted-motion, or native admission."}
    return raw, proof
