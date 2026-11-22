"""Read registered NHTISS4 material frames and named muscle attachment sites."""

import hashlib
import struct

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tissue(data, sid):
    magic, version, nr, nb, nv, ni, _ = struct.unpack_from("<8s6I", data)
    if (magic, version) != (b"NHTISS4\0", 5):
        raise ValueError("Unsupported muscle payload")
    records = np.frombuffer(data, "<u4", 8 * nr, 64).reshape(nr, 8)
    (row,) = records[records[:, 6] == sid]
    bi, nbind, vi, count, fi, nidx, *_ = map(int, row)
    vo = 64 + 32 * nr + 36 * nb
    source_faces = np.frombuffer(data, "<u4", nidx, vo + 56 * nv + 4 * fi).reshape(
        -1, 3
    )
    ids = np.unique(source_faces)
    if ids[0] < vi or ids[-1] >= vi + count:
        raise ValueError("Invalid muscle allocation")
    verts = np.frombuffer(data, "<f4", 14 * nv, vo).reshape(nv, 14)[ids]
    bindings = [
        struct.unpack_from("<I8f", data, 64 + 32 * nr + 36 * (bi + i))
        for i in range(nbind)
    ]
    slots = verts[:, 6:10].copy().view("<u4")
    weights = verts[:, 10:14].astype(float)
    if (
        (slots >= nbind).any()
        or (weights < 0).any()
        or np.abs(weights.sum(1) - 1).max() > 1e-5
    ):
        raise ValueError("Unsupported native sparse weights")
    return (
        verts[:, :3].astype(float),
        np.searchsorted(ids, source_faces),
        slots,
        weights,
        bindings,
        ids,
        vo,
    )


def landmarks(bindings, poses, routes):
    by_body = {r[0]: r for r in bindings}
    original, target, groups = [], [], []
    for route in routes:
        for s in route:
            b = s["body"]
            q = by_body[b]
            local = np.asarray(s["local_m"])
            original.append(
                Rotation.from_quat(q[4:8]).inv().apply((local - q[1:4]) / q[8])
            )
            p = poses[b]
            target.append(
                Rotation.from_quat(p["quaternion_xyzw"]).apply(local) + p["position_m"]
            )
            groups.append(b)
    return np.array(original), np.array(target), np.array(groups)


def affines(slots, w, bindings, poses):
    matrices = []
    shifts = []
    for q in bindings:
        p = poses[q[0]]
        r = Rotation.from_quat(p["quaternion_xyzw"])
        matrices.append((r * Rotation.from_quat(q[4:8])).as_matrix() * q[8])
        shifts.append(r.apply(q[1:4]) + p["position_m"])
    A = np.einsum("vb,vbij->vij", w, np.asarray(matrices)[slots])
    t = np.einsum("vb,vbi->vi", w, np.asarray(shifts)[slots])
    return A, t


def apply(A, x):
    return np.einsum("vij,vj->vi", A, x)
