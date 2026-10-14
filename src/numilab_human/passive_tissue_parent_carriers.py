"""Encode inferred passive reference weights through the existing NHTISS parent path.

These load-only carriers are not additional anatomical surfaces or physical
state. Their vertices retain the child's source coordinates and normals; each
carrier has one body influence. The existing parent loader compiles the full
positive field once, then accepted articulated poses drive it on Metal.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CarrierRow:
    stable_id: int
    vertices6: np.ndarray
    faces: np.ndarray
    binding_bytes: bytes
    weights: np.ndarray
    layer: int = 1



def validate_compiled_core_weights(
    source_gids, source_weights_float64, source_core_body_indices,
    compiled_gids, compiled_weights, compiled_core_body_indices,
) -> dict:
    """Prove native Float32 casts by source GID without normalizing either field.

    Source doubles obey the existing 1e-12 partition guard. Compiled floats,
    including exact promotion back to doubles for diagnostics, are compared
    to their source casts; their row sums are not reinterpreted as source data.
    """
    from .resting_anatomy_verify import keyed_row_indices

    source = np.asarray(source_weights_float64)
    compiled = np.asarray(compiled_weights)
    source_ids = np.asarray(source_gids)
    compiled_ids = np.asarray(compiled_gids)
    source_bodies = tuple(source_core_body_indices)
    compiled_bodies = tuple(compiled_core_body_indices)
    if (not source_bodies or len(source_bodies) > 24
            or any(isinstance(body, (bool, np.bool_)) or not isinstance(body, (int, np.integer))
                   or body < 0 for body in source_bodies)
            or len(set(source_bodies)) != len(source_bodies)
            or compiled_bodies != source_bodies):
        raise ValueError("ordered core-body identities differ or are invalid")
    if (source.dtype != np.dtype("float64")
            or source.shape != (len(source_ids), len(source_bodies))
            or not len(source_ids) or not np.isfinite(source).all()
            or (source < 0).any() or (source > 1).any()
            or np.max(np.abs(source.sum(axis=1) - 1)) > 1e-12):
        raise ValueError("source Float64 weights must be a normalized partition")
    if (compiled.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or compiled.shape != (len(compiled_ids), len(source_bodies))
            or not len(compiled_ids) or not np.isfinite(compiled).all()):
        raise ValueError("compiled weights must be aligned Float32 or exact promoted Float64")
    # Validate both complete ID inventories before selecting the active subset.
    keyed_row_indices(source_ids, source_ids)
    keyed_row_indices(compiled_ids, compiled_ids)
    rows = keyed_row_indices(source_ids, compiled_ids)
    expected = source[rows].astype(np.float32).astype(compiled.dtype)
    if compiled.tobytes() != expected.tobytes():
        raise ValueError("compiled weights differ from exact keyed source Float32 casts")
    return {
        "source_rows": len(source_ids), "compiled_rows": len(compiled_ids),
        "ordered_core_body_indices": [int(body) for body in source_bodies],
        "source_partition_max_error": float(np.max(np.abs(source.sum(axis=1) - 1))),
        "compiled_partition_max_error_diagnostic":
            float(np.max(np.abs(compiled.astype(np.float64).sum(axis=1) - 1))),
        "compiled_rows_equal_keyed_source_float32_casts": True,
        "source_guard_unchanged": True, "weights_modified": False,
    }

def encode_carriers(
    rows: list[CarrierRow], *, registration_fingerprint: int, source_digest: bytes
) -> tuple[bytes, list[dict], dict]:
    """Return NHTISS4 ABI5 bytes and existing source-parent surface terms.

    Input weights must already be a nonnegative partition. No top-k truncation,
    field fitting, normalization, or anatomical qualification occurs here.
    Source index zero in the returned terms refers to the returned payload.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(type(registration_fingerprint) is int
            and 0 < registration_fingerprint < 2**32, "invalid registration fingerprint")
    require(isinstance(source_digest, bytes) and len(source_digest) == 32,
            "invalid physical source digest")
    require(0 < len(rows) <= 192, "invalid carrier row count")
    records, bindings, vertices, indices, surfaces = [], [], [], [], []
    seen = set()
    point_fields = {}
    point_bytes = {}
    nb = nv = ni = 0
    max_error = 0.0
    for row in rows:
        require(type(row.stable_id) is int and 0 < row.stable_id < 2**32
                and row.stable_id not in seen, "invalid or repeated stable ID")
        seen.add(row.stable_id)
        require(row.layer in (1, 2, 3), "invalid carrier layer")
        v = np.asarray(row.vertices6, dtype="<f4")
        raw_faces = np.asarray(row.faces)
        require(raw_faces.dtype.kind in "iu", "faces must use integer indices")
        f = raw_faces.astype(np.int64)
        w = np.asarray(row.weights, dtype=np.float64)
        require(v.ndim == 2 and v.shape[1] == 6 and len(v) > 0
                and np.isfinite(v).all(), "invalid source vertex fields")
        require(np.max(np.abs(np.linalg.norm(v[:, 3:].astype(float), axis=1)-1)) <= 2e-3,
                "source normals must be unit length")
        require(f.ndim == 2 and f.shape[1] == 3 and len(f) > 0
                and f.min() >= 0 and f.max() < len(v), "invalid source faces")
        # Validate the coordinates actually serialized, after Float32 rounding.
        # A nonzero rational/Float64 source triangle can collapse in NHTISS.
        # Reuse the exact dyadic predicate; no geometric tolerance or repair.
        from .cardiac_cavity_intersections import _records, float32_point_lattice_key
        _records([float32_point_lattice_key(tuple(float(x) for x in point))
                  for point in v[:, :3]], f.tolist())
        require(isinstance(row.binding_bytes, bytes) and len(row.binding_bytes) % 36 == 0,
                "invalid binding byte count")
        require(row.binding_bytes == rows[0].binding_bytes,
                "paired carriers require one canonical binding table")
        bc = len(row.binding_bytes)//36
        require(1 <= bc <= 24 and w.shape == (len(v), bc),
                "weights do not match source vertices and bindings")
        require(np.isfinite(w).all() and (w >= 0).all() and (w <= 1).all(),
                "weights must be finite and bounded")
        require(np.max(np.abs(w.sum(axis=1)-1)) <= 1e-12,
                "weights must form a partition of unity")
        for point, field in zip(v[:, :3], w):
            key = tuple(float(x) for x in point)
            require(point_bytes.setdefault(key, point.tobytes()) == point.tobytes(),
                    "coincident source coordinates differ in stored bits")
            bits = field.tobytes()
            require(point_fields.setdefault(key, bits) == bits,
                    "coincident source points have different shared fields")
        bodies = []
        for offset in range(0, len(row.binding_bytes), 36):
            core, *values = struct.unpack_from("<I8f", row.binding_bytes, offset)
            require(core != 2**32-1 and core not in bodies, "invalid or repeated body binding")
            bodies.append(core)
            require(np.isfinite(values).all() and values[7] > 0
                    and abs(np.linalg.norm(values[3:7])-1) <= 2e-3,
                    "invalid registered binding transform")
        source_vertices, body_columns = np.nonzero(w > 0)
        count = len(source_vertices)
        require(nv+count <= 1_000_000 and ni+f.size <= 6_000_000,
                "carrier exceeds existing NHTISS bounds")
        coefficients = w[source_vertices, body_columns]
        compiled = coefficients.astype("<f4")
        require((compiled > 0).all(), "positive parent coefficient underflows Float32")
        max_error = max(max_error, float(np.max(np.abs(compiled.astype(float)-coefficients))))
        packed = np.zeros(count, dtype=np.dtype([
            ("vertex", "<f4", (6,)), ("binding", "<u4", (4,)),
            ("weight", "<f4", (4,))]))
        require(packed.dtype.itemsize == 56, "NHTISS vertex stride changed")
        packed["vertex"] = v[source_vertices]
        packed["binding"][:, 0] = body_columns
        packed["weight"][:, 0] = 1
        counts = np.count_nonzero(w > 0, axis=1)
        first = np.concatenate(([0], np.cumsum(counts[:-1])))
        vertex_terms = [
            [[0, row.stable_id, int(k), float(coefficients[k])]
             for k in range(int(start), int(start+n))]
            for start, n in zip(first, counts)
        ]
        records.append(struct.pack("<8I", nb, bc, nv, count, ni, f.size,
                                   row.stable_id, row.layer))
        bindings.append(row.binding_bytes)
        vertices.append(packed.tobytes())
        # Carrier faces retain the source indexing on one representative copy.
        # The extra copies exist only to carry positive body influences at load.
        indices.append((first[f]+nv).astype("<u4").tobytes())
        surfaces.append({"stable_id": row.stable_id, "vertices": vertex_terms})
        nb += bc
        nv += count
        ni += f.size
    require(nv <= 4_194_304, "compiled influences exceed existing parent-load budget")
    header = struct.pack("<8s6I32s", b"NHTISS4\0", 5, len(rows), nb, nv, ni,
                         registration_fingerprint, source_digest)
    payload = header+b"".join(records+bindings+vertices+indices)
    require(len(payload) <= 128*1024*1024, "carrier exceeds parent payload byte bound")
    return payload, surfaces, {
        "carrier_rows": len(rows), "source_vertices": nv,
        "compiled_positive_influences": nv,
        "maximum_weight_float32_rounding_error": max_error,
        "runtime_format": "existing NHTISS4 ABI5 and passive_tissue_parent_bindings",
        "scope": "Inferred load-only passive field carrier; no physical state, forces, or anatomy admission.",
    }


def _retarget_carrier_binding_bytes(
    raw: bytes, *, previous_fingerprint: int, fingerprint: int,
    source_digest: bytes, previous_bindings: dict[int, bytes],
    bindings: dict[int, bytes], coordinate_frame: str,
) -> tuple[bytes, dict]:
    """Re-emit an existing carrier under its verified source coordinate frame."""
    import hashlib

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(coordinate_frame in ("registered_source_atlas", "core_inertial_local"),
            "unsupported carrier coordinate frame")
    require(len(raw) >= 64, "truncated carrier")
    magic, abi, nr, nb, nv, ni, old_fp, digest = struct.unpack_from("<8s6I32s", raw)
    require(magic == b"NHTISS4\x00" and abi == 5 and 0 < nr <= 192
            and len(raw) == 64 + nr*32 + nb*36 + nv*56 + ni*4,
            "invalid existing carrier ABI")
    require(old_fp == previous_fingerprint and digest == source_digest
            and type(fingerprint) is int and 0 < fingerprint < 2**32,
            "carrier source identity or registration differs")
    require(set(previous_bindings) == set(bindings), "carrier Core identity map differs")
    start = 64 + nr*32
    out = bytearray(raw)
    changes = []
    for ordinal in range(nb):
        offset = start + ordinal*36
        record = raw[offset:offset+36]
        core = struct.unpack_from("<I", record)[0]
        require(core in previous_bindings, "carrier Core index is absent from registration")
        if coordinate_frame == "registered_source_atlas":
            require(record == previous_bindings[core],
                    "carrier binding differs from its exact previous registration")
            new_record = bindings[core]
        else:
            identity = struct.pack("<I8f", core, 0, 0, 0, 0, 0, 0, 1, 1)
            require(record == identity, "Core-local carrier binding is not identity")
            new_record = record
        require(len(new_record) == 36 and struct.unpack_from("<I", new_record)[0] == core,
                "new carrier binding changes Core identity")
        values = struct.unpack_from("<I8f", new_record)[1:]
        require(np.isfinite(values).all() and values[-1] > 0
                and abs(np.linalg.norm(values[3:7])-1) <= 2e-3,
                "new carrier binding transform is invalid")
        out[offset:offset+36] = new_record
        if record != new_record:
            changes.append({"binding_ordinal": ordinal, "core_body_index": core,
                            "previous_sha256": hashlib.sha256(record).hexdigest(),
                            "sha256": hashlib.sha256(new_record).hexdigest()})
    out[:64] = struct.pack("<8s6I32s", magic, abi, nr, nb, nv, ni, fingerprint, digest)
    require(out[64:start] == raw[64:start]
            and out[start+nb*36:] == raw[start+nb*36:],
            "carrier re-emission changed geometry or influence state")
    return bytes(out), {
        "coordinate_frame": coordinate_frame, "carrier_rows": nr,
        "binding_count": nb, "vertex_count": nv, "index_count": ni,
        "previous_registration_fingerprint32": f"{old_fp:08x}",
        "registration_fingerprint32": f"{fingerprint:08x}",
        "changed_binding_records": changes,
        "geometry_normals_weights_indices_and_rows_byte_exact": True,
        "source_terms_not_modified": True,
        "scope": "Existing load-only carrier registration preparation; no new physical state or anatomy admission.",
    }


def reemit_registered_carrier(
    payload, previous_registration, registration, artifact, output, *,
    coordinate_frame: str,
) -> dict:
    """Validate both actual rigid rest frames before carrier registration re-emission."""
    import hashlib
    import json
    from pathlib import Path

    from .model import _bodyparts_runtime_bindings, _bodyparts_visual_local_pose
    from .passive_attachment_composition import _read_nhtiss4

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    payload, previous_registration, registration, artifact, output = (
        Path(p).resolve() for p in
        (payload, previous_registration, registration, artifact, output))
    if output.exists():
        raise ValueError("carrier output already exists")
    old_doc = json.loads(previous_registration.read_text())
    doc = json.loads(registration.read_text())
    old_context, old_bodies = _bodyparts_runtime_bindings(old_doc, artifact)
    context, bodies = _bodyparts_runtime_bindings(doc, artifact)
    if (old_context["rigid"]["sha256"] != context["rigid"]["sha256"]
            or old_bodies != bodies
            or old_doc["source"]["bodyparts"] != doc["source"]["bodyparts"]):
        raise ValueError("carrier registration changes rigid frames or source anatomy")

    def binding_map(document):
        result = {}
        names = {}
        for anchor in document["anchors"]:
            core = anchor["target"]["core_body_index"]
            name = anchor["target"]["name"]
            translation, quaternion, scale = _bodyparts_visual_local_pose(
                anchor["registration"]["source_obj_mm_to_core_inertial_body_m"],
                f"carrier Core {core} registration")
            record = struct.pack("<I8f", core, *translation, *quaternion, scale)
            if core in result and (result[core] != record or names[core] != name):
                raise ValueError("carrier registration aliases one Core frame")
            result[core], names[core] = record, name
        return result, names

    old_bindings, old_names = binding_map(old_doc)
    bindings, names = binding_map(doc)
    if old_names != names:
        raise ValueError("carrier registration changes anatomical Core identities")
    data = _read_nhtiss4(payload)
    raw, report = _retarget_carrier_binding_bytes(
        data["raw"], previous_fingerprint=int(sha(previous_registration)[:8], 16),
        fingerprint=int(sha(registration)[:8], 16),
        source_digest=bytes.fromhex(context["source_archive_sha256"]),
        previous_bindings=old_bindings, bindings=bindings,
        coordinate_frame=coordinate_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    emitted = _read_nhtiss4(output)
    if emitted["raw"] != raw:
        raise ValueError("carrier output differs from verified owner emission")
    report.update(
        source_payload_path=str(payload), source_payload_sha256=data["sha256"],
        output_payload_path=str(output), output_payload_sha256=sha(output),
        previous_registration_path=str(previous_registration),
        previous_registration_sha256=sha(previous_registration),
        registration_path=str(registration), registration_sha256=sha(registration),
        actual_rigid_payload_sha256=context["rigid"]["sha256"],
        owner_source_sha256=sha(Path(__file__)),
        anatomical_and_native_admission=False)
    return report
