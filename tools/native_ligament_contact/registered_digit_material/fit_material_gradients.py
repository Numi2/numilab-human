"""Integrate the registered donor rotation field over actual skin/fat tetrahedra.

This is a reference-pose preparation problem. The shape and proximity
objective guides mesh registration only and supplies no forces to simulation.
"""

import csv, json, re, sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from scipy.optimize import nnls
from fit_dorsal_material import inputs
from prepare_connected_pose import read, sha, validate
from audit_settled_material_placement import transport_material_nodes


def field_system(x, cells, roles, owner, rot, modulus, protect_slivers):
    n = len(x)
    K = csr_matrix((n, n))
    rhs = np.zeros_like(x)
    mass = np.zeros(n)
    for first in range(0, len(cells), 49152):
        c = cells[first : first + 49152]
        E = x[c[:, 1:]] - x[c[:, 0, None]]
        V = np.linalg.det(E) / 6
        if (V <= 0).any():
            raise ValueError("Invalid input volume")
        native_weight = V * modulus[roles[first : first + len(c)]]
        v = native_weight
        if protect_slivers:
            # Per-element shape protection is an authoring metric. Slivers
            # have little volume but must still meet the same gradient gate.
            other = E[:, [1, 2, 2]] - E[:, [0, 0, 1]]
            mean_length_sq = (
                (E * E).sum(axis=(1, 2)) + (other * other).sum(axis=(1, 2))
            ) / 6
            regular_volume = mean_length_sq**1.5 / (6 * np.sqrt(2.0))
            v = (
                np.maximum(V, 0.05 * regular_volume)
                * modulus[roles[first : first + len(c)]]
            )

        grads = np.empty((len(c), 4, 3))
        grads[:, 1:] = np.linalg.inv(E).transpose(0, 2, 1)
        grads[:, 0] = -grads[:, 1:].sum(axis=1)
        A = v[:, None, None] * (grads @ grads.transpose(0, 2, 1))
        ii = np.broadcast_to(c[:, :, None], A.shape).ravel()
        jj = np.broadcast_to(c[:, None, :], A.shape).ravel()
        K += csr_matrix((A.ravel(), (ii, jj)), shape=(n, n))
        r = rot[owner[c]].mean(axis=1)
        U, _, Vh = np.linalg.svd(r)
        U[:, :, -1] *= np.sign(np.linalg.det(U @ Vh))[:, None]
        rigid = U @ Vh
        load = v[:, None, None] * (grads @ (rigid - np.eye(3)).transpose(0, 2, 1))
        for k in range(4):
            np.add.at(mass, c[:, k], 0.25 * native_weight)
            for j in range(3):
                np.add.at(rhs[:, j], c[:, k], load[:, k, j])
    return K, rhs, mass


def unilateral_z(factor, free, base, outer):
    # Compute the Schur complement only on witnesses that actually violate
    # the fixed flat bed. Include newly encountered witnesses until the full
    # outer surface is feasible, and verify multipliers/complementarity.
    local = np.full(len(base), -1, dtype=int)
    local[free] = np.arange(len(free))
    ids = np.array([], dtype=int)
    Z = np.empty((0, 0))
    height = base.copy()
    loads = np.array([])
    for iteration in range(18):
        bad = outer[(height[outer] < -2.0e-9) & ~np.isin(outer, ids)]
        if not len(bad):
            break
        if (local[bad] < 0).any():
            raise ValueError("A hard anatomical attachment is below the fixed bed")
        # Most negative new witnesses first. No relaxation of the final all-node test.
        new = bad[np.argsort(height[bad])[:96]]
        n = len(ids)
        ids = np.r_[ids, new]
        if len(ids) > 1200:
            raise RuntimeError("Unbounded bed active set")
        M = np.empty((len(ids), len(ids)))
        M[:n, :n] = Z
        for k in range(n, len(ids), 8):
            end = min(k + 8, len(ids))
            Q = np.zeros((len(free), end - k), order="F")
            Q[local[ids[k:end]], np.arange(end - k)] = 1
            M[:, k:end] = factor.solve(Q)[local[ids]]
        M[n:, :n] = M[:n, n:].T
        Z = (M + M.T) / 2
        L = np.linalg.cholesky(Z)
        loads = nnls(
            L.T, np.linalg.solve(L, -base[ids]), maxiter=max(100, 6 * len(ids))
        )[0]
        f = np.zeros(len(free))
        f[local[ids]] = loads
        height = base.copy()
        height[free] += factor.solve(f)
    if height[outer].min() < -2e-9:
        raise ValueError(
            f"Unilateral registration solve missed {height[outer].min()} m"
        )
    # Algebraically active Dirichlet coordinates equal the plane. Set exact
    # zero for those solved multipliers before Float32 conversion.
    if len(ids):
        active = loads > 0
        if active.any() and np.max(np.abs(height[ids[active]])) > 2e-8:
            raise ValueError("Unclosed active complementarity")
        height[ids[active]] = 0.0
    return height, dict(
        candidate_witnesses=len(ids),
        active=int((loads > 0).sum()),
        worst_gap_m=float(height[outer].min()),
    )


def fit(fine, compact, material, bed, layers, seed, out, protect_slivers=False):
    out.mkdir(exist_ok=False)
    ref, start, cells, roles, pins = inputs(fine)
    x = start.astype(float)
    ref = ref.astype(float)
    report = read(seed / "quality.json")
    if (
        report["candidate_sha256"] != sha(seed / "positions.npy")
        or pins != report["source"]
        or sha(bed) != report["attachment_registration"]
    ):
        raise ValueError("Mismatched input")
    first = fine / "native/accepted-geometry/step-1.receipt.json"
    end = compact / "terminal.receipt.json"
    if report["opening"] != sha(first) or report["target"] != sha(end):
        raise ValueError("Pose input changed")
    opening = {
        r["body_index"]: r
        for r in read(first)["initial_anatomical_registration"]["body_poses"]
    }
    target = {r["body_index"]: r for r in read(end)["accepted_registered_body_poses"]}
    cm = Path(
        read(fine / "launch.json")["environment"][
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV"
        ]
    )
    if report["ancestry"] != sha(cm):
        raise ValueError("Invalid ancestry")
    owner = np.full(len(x), -1, dtype=int)
    with cm.open() as f:
        for r in csv.DictReader(f):
            i = int(r["joint_node_id"]) - 1
            b = int(r["donor_body"])
            if owner[i] not in (-1, b):
                raise ValueError("Contradictory ancestry")
            owner[i] = b
    if np.any(owner < 0):
        raise ValueError("Missing ancestry")
    rot = np.empty((max(opening) + 1, 3, 3))
    for b in np.unique(owner):
        rot[b] = (
            Rotation.from_quat(target[b]["quaternion_xyzw"])
            * Rotation.from_quat(opening[b]["quaternion_xyzw"]).inv()
        ).as_matrix()
    params = read(material / "fit.json")["source_materials"]
    mu = np.zeros(17)
    for name, h in params.items():
        path = Path(name)
        if sha(path) != h:
            raise ValueError("Stale material calibration")
        idx = 15 if "dermis" in path.name else 16
        mu[idx] = 1e6 * float(
            re.search(
                r"parameter\s+shear\s*:\s*MPa\s*=\s*([\d.eE+-]+)", path.read_text()
            )[1]
        )
    if set(roles) != {15, 16} or min(mu[15:]) <= 0:
        raise ValueError("Unexpected material")
    anchors = read(bed)["attachment_rows"]
    aid = np.array([r["volume_row"] for r in anchors])
    bounds = np.load(layers)["Orows"]
    approx = np.load(seed / "positions.npy").astype(float)
    direct = np.array(
        [
            transport_material_nodes(x[i : i + 1], opening[owner[i]], target[owner[i]])[
                0
            ]
            for i in aid
        ]
    )
    h = report["grid_spacing_m"] / 2  # screen over half the observed spatial bin.
    K, rhs, lumped = field_system(x, cells, roles, owner, rot, mu, protect_slivers)
    screen = lumped / h**2
    K += diags(screen)
    rhs += screen[:, None] * (approx - x)
    free = np.setdiff1d(np.arange(len(x)), aid)
    rhs_free = rhs[free] - K[free][:, aid] @ (direct - x[aid])
    reduced = K[free][:, free].tocsc()
    del K
    print("factoring", reduced.shape, reduced.nnz, flush=True)
    f = splu(
        reduced,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True},
    )
    sol = f.solve(rhs_free)
    rel = float(np.linalg.norm(reduced @ sol - rhs_free) / np.linalg.norm(rhs_free))
    if not np.isfinite(rel) or rel > 1e-7:
        raise ValueError(f"Uncertified linear reconstruction {rel}")
    trial = x.copy()
    trial[free] += sol
    trial[aid] = direct
    print("unconstrained bed", float(trial[bounds, 2].min()), flush=True)
    trial[:, 2], contact = unilateral_z(f, free, trial[:, 2], bounds)
    trial = trial.astype("<f4").astype(float)
    actual = validate(ref, trial, cells, roles)
    result = dict(
        sliver_protection=protect_slivers,
        seed_sha256=report["candidate_sha256"],
        material_sha256=params,
        source=pins,
        screen_length_m=h,
        linear_residual=rel,
        hard_attachment_count=len(aid),
        max_attachment_error_m=float(np.linalg.norm(trial[aid] - direct, axis=1).max()),
        bed=contact
        | dict(outer_nodes=len(bounds), below=int(np.sum(trial[bounds, 2] < 0))),
        cells=actual,
        rms_seed_error_m=float(np.sqrt(np.mean(np.sum((trial - approx) ** 2, axis=1)))),
        scope="Offline strain-aware placement with hard anatomical attachment positions and unilateral bed inequalities. The registration multipliers are not physical reaction measurements.",
    )
    np.save(out / "positions.npy", trial.astype("<f4"))
    result["candidate_sha256"] = sha(out / "positions.npy")
    (out / "quality.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    fit(
        *(Path(s).resolve() for s in sys.argv[1:8]),
        protect_slivers="--protect-slivers" in sys.argv[8:],
    )
