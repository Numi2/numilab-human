"""Construct a continuous material pose by fitting rotations on the bonded tetrahedra.

This is an authoring solve. Its objective and attachment constraints exist only
while constructing positions. Admission uses the unmodified material and bed.
"""

from pathlib import Path
import csv, json, re, sys
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from supported_material_admission import geom
from live_pose_frames import current_donors
from compile_connected_joint import digest


def project(case, volume, skeleton, support, bindings, out):
    rig = json.loads(skeleton.read_bytes())
    if (
        rig["current_launch_sha256"] != digest(case / "launch.json")
        or rig["final_joint_limit_violations"]
    ):
        raise ValueError("Unmatched or out-of-bounds skeleton")
    dst = np.asarray(rig["target_poses"])
    size = len(dst)
    d = np.zeros((size, 3, 3))
    p = np.zeros((size, 3))
    present = set()
    live, donors = current_donors(case)
    for src in donors:
        k = src["body_index"]
        cur = src["current"]
        present.add(k)
        d[k] = (
            Rotation.from_quat(dst[k, 3:])
            * Rotation.from_quat(cur["orientation_world_xyzw"]).inv()
        ).as_matrix()
        p[k] = dst[k, :3] - d[k] @ cur["position_world_m"]
    read = lambda n: np.loadtxt(volume / n, skiprows=1, usecols=(1, 2, 3))
    y = read("current.node")
    x = read("mesh.node")
    T = (
        np.loadtxt(volume / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4), dtype="i4")
        - 1
    )
    label = np.loadtxt(
        volume / "current-child.labels.csv",
        skiprows=1,
        delimiter=",",
        usecols=(0, 2),
        dtype="i4",
    )
    owner = label[:, 1]
    if not np.array_equal(label[:, 0], np.arange(len(T))) or not set(owner).issubset(
        present
    ):
        raise ValueError("Unmatched source ownership")
    cell = np.fromfile(volume / "cell-mass.f64", "<f8")
    node = np.fromfile(volume / "node-mass.f64", "<f8")
    if (
        cell.shape != (len(T),)
        or node.shape != (len(y),)
        or np.any(node <= 0)
        or np.any(cell <= 0)
    ):
        raise ValueError("Invalid mass")
    # Estimate each owner's length scale from its registered material second moment.
    total = np.bincount(owner, weights=cell, minlength=size)
    center = np.zeros((size, 3))
    rad = np.zeros(size)
    chunks = np.array_split(np.arange(len(T)), 48)
    for g in chunks:
        np.add.at(center, owner[g], cell[g, None] * y[T[g]].mean(1))
    nz = total > 0
    center[nz] /= total[nz, None]
    for g in chunks:
        r = y[T[g]] - center[owner[g], None, :]
        # Integral of squared radius over a uniform tetrahedron, including linear terms.
        ms = (np.sum(r * r, axis=(1, 2)) + np.sum(r.sum(1) ** 2, axis=1)) / 20.0
        np.add.at(rad, owner[g], cell[g] * ms)
    rad[nz] /= total[nz]
    if np.any(rad[nz] <= 0):
        raise ValueError("Degenerate material extent")
    eye = np.eye(3)
    rhs = np.zeros_like(y)
    replay = np.zeros_like(node)
    K = diags(node).tocsr()
    for g in chunks:
        t = T[g]
        o = owner[g]
        yy = y[t]
        inv = np.linalg.inv(yy[:, 1:] - yy[:, :1])
        grad = np.concatenate((-inv.sum(2)[:, :, None], inv), axis=2).transpose(0, 2, 1)
        if np.abs(np.einsum("nai,naj->nij", grad, yy - yy[:, :1]) - eye).max() > 5e-8:
            raise ValueError("Incomplete affine gradient")
        w = cell[g] * rad[o]
        local = w[:, None, None] * (grad @ grad.transpose(0, 2, 1))
        row, col = np.broadcast_arrays(t[:, :, None], t[:, None, :])
        K += coo_matrix(
            (local.ravel(), (row.ravel(), col.ravel())), shape=K.shape
        ).tocsr()
        delta = d[o] - eye
        proposal = np.einsum("nij,nkj->nki", delta, yy) + p[o, None, :]
        for j in range(4):
            np.add.at(
                rhs,
                t[:, j],
                cell[g, None] * 0.25 * proposal[:, j]
                + w[:, None] * np.einsum("nij,nj->ni", delta, grad[:, j]),
            )
            np.add.at(replay, t[:, j], cell[g] * 0.25)
    if np.max(abs(replay - node) / node) > 1e-8:
        raise ValueError("Cell shares do not match native nodal masses")
    anchors = re.findall(r"^\{\d+u,(\d+)u,(\d+)u,\{", bindings.read_text(), re.M)
    fixed = np.array([int(i) for i, _ in anchors])
    donors = np.array([int(j) for _, j in anchors])
    if (
        len(fixed) != 14
        or len(set(fixed)) != len(fixed)
        or not set(donors).issubset(present)
    ):
        raise ValueError("Unmatched attachment ownership")
    u = np.zeros_like(y)
    u[fixed] = np.einsum("nij,nj->ni", d[donors] - eye, y[fixed]) + p[donors]
    free = np.ones(len(y), bool)
    free[fixed] = False
    rhs -= K @ u
    A = K[free][:, free].tocsc()
    del K, grad, yy, local, label, replay
    scale = 1 / np.sqrt(A.diagonal())
    fact = splu(
        (diags(scale) @ A @ diags(scale)).tocsc(),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0,
    )
    u[free] = scale[:, None] * fact.solve(scale[:, None] * rhs[free])
    residual = np.linalg.norm(A @ u[free] - rhs[free]) / np.linalg.norm(rhs[free])
    if residual > 1e-6:
        raise ValueError("Inaccurate compatibility solve")
    del fact, A
    bed = json.loads(support.read_bytes())
    if bed["launch_sha256"] != rig["current_launch_sha256"]:
        raise ValueError("Unmatched bed")
    n = np.asarray(bed["bed_normal"])
    off = bed["source_plane_offset_m"]
    outer = np.unique(np.load(volume / "material-walls.npz")["O"])
    ref = np.concatenate([geom(x, T[g])[0] for g in chunks])
    np.save(out.with_suffix(".displacement.npy"), u, allow_pickle=False)
    trials = []
    for alpha in (0.0, 0.001, 0.01, 0.05, 0.25, 1.0):
        displacement = (y + alpha * u) - x
        high = displacement.astype("f4").astype("f8")
        low = (displacement - high).astype("f4").astype("f8")
        z = x + (high + low)
        if alpha == 0 and not np.array_equal(z, y):
            raise ValueError("Invalid baseline")
        v, h = np.concatenate([np.stack(geom(z, T[g])) for g in chunks], axis=1)
        J = v / ref
        bad = (v <= 1.0e-18) | (J < 0.4) | (J > 2.5) | (h < 1.0e-5)
        gaps = z[outer] @ n - off
        trials.append(
            dict(
                fraction=alpha,
                invalid_cells=int(bad.sum()),
                inverted=int((v <= 0).sum()),
                failed_shape=int((h < 1e-5).sum()),
                min_J=float(J.min()),
                max_J=float(J.max()),
                outer_nodes_below_bed=int((gaps < 0).sum()),
                min_surface_gap_m=float(gaps.min()),
            )
        )
    info = dict(
        reference=digest(volume / "mesh.node"),
        registered=digest(volume / "current.node"),
        labels=digest(volume / "current-child.labels.csv"),
        cell_mass=digest(volume / "cell-mass.f64"),
        nodal_mass=digest(volume / "node-mass.f64"),
        bed=digest(support),
        live_body_pose=digest(live),
        skeleton=digest(skeleton),
        attachment_source=digest(bindings),
        registered_attachments=len(fixed),
        residual=residual,
        trials=trials,
    )
    out.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    print([(v["fraction"], v["invalid_cells"], v["inverted"]) for v in trials])


if __name__ == "__main__":
    project(*(Path(s).resolve() for s in sys.argv[1:]))
