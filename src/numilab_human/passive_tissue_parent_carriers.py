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
