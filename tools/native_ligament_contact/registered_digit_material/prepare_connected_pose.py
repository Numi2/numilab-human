"""Offline registration candidate for the dense native skin/fat mesh in a measured supported posture."""

import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator
from fit_dorsal_material import inputs
from audit_settled_material_placement import transport_material_nodes
from torso_cell_geometry import geom


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def validate(ref, trial, tets, roles):
    result = {}
    for material in np.unique(roles):
        cells = tets[roles == material]
        bad = inv = strain = 0
        jmin = np.inf
        jmax = -np.inf
        fmin = np.inf
        fmax = -np.inf
        stretch_fail = 0
        ids = []
        for j in range(0, len(cells), 32768):
            c = cells[j : j + 32768]
            vol, quality = geom(trial, c)
            rest, _ = geom(ref, c)
            J = vol / rest
            edges = trial[c[:, 1:]] - trial[c[:, 0, None]]
            old = ref[c[:, 1:]] - ref[c[:, 0, None]]
            grad = np.linalg.solve(old, edges)
            singular = np.linalg.svd(grad, compute_uv=False)
            lo = singular[:, -1]
            hi = singular[:, 0]
            jac = (J < 0.4) | (J > 2.5)
            shape = (vol <= 1e-18) | (quality < 1e-5)
            bad += int(shape.sum())
            inv += int((vol <= 0).sum())
            strain += int(jac.sum())
            # Stored as engineering diagnostics; native owner still performs its complete admission.
            stretch_fail += int(((lo < 0.5) | (hi > 2)).sum())
            jmin = min(jmin, float(J.min()))
            jmax = max(jmax, float(J.max()))
            fmin = min(fmin, float(lo.min()))
            fmax = max(fmax, float(hi.max()))
            if len(ids) < 16:
                ids.extend((j + np.flatnonzero(shape | jac)[: 16 - len(ids)]).tolist())
        result[int(material)] = dict(
            count=len(cells),
            inverted=inv,
            shape_or_volume_failure=bad,
            volume_ratio_outside_0_4_to_2_5=strain,
            principal_stretch_outside_0_5_to_2=stretch_fail,
            relative_volume=[jmin, jmax],
            principal_stretch=[fmin, fmax],
            first_failed_material_local_rows=ids,
        )
    return result


def prep(fine, compact, bed, layer, calibration, out):
    out.mkdir(exist_ok=False)
    ref, start, tets, role, pins = inputs(fine)
    ref = ref.astype(float)
    x = start.astype(float)
    first = fine / "native/accepted-geometry/step-1.receipt.json"
    endfile = compact / "terminal.receipt.json"
    f = read(first)
    e = read(endfile)
    c = read(calibration)
    if (
        f["physical_endpoint"] != "accepted"
        or f["accepted_step"] != 1
        or e["accepted_step"] != 30000
        or e["physical_endpoint"] != "accepted"
    ):
        raise ValueError(
            "Expected accepted dense opening and supported 60-second destination"
        )
    opening = {
        r["body_index"]: r for r in f["initial_anatomical_registration"]["body_poses"]
    }
    target = {r["body_index"]: r for r in e["accepted_registered_body_poses"]}
    if set(opening) != set(target):
        raise ValueError("Body inventory differs")
    cm = Path(
        read(fine / "launch.json")["environment"][
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV"
        ]
    )
    if (
        c["dense_map"] != sha(cm)
        or c["fine"] != sha(fine / "launch.json")
        or c["compact"] != sha(compact / "launch.json")
    ):
        raise ValueError("Stale material/pose pair")
    owner = np.full(len(x), -1, dtype=int)
    with cm.open() as stream:
        for row in csv.DictReader(stream):
            i = int(row["joint_node_id"]) - 1
            body = int(row["donor_body"])
            if not 0 <= i < len(x) or body not in opening or owner[i] not in (-1, body):
                raise ValueError("Invalid body ancestry")
            owner[i] = body
    if (owner < 0).any():
        raise ValueError("Unowned material nodes")
    direct = np.empty_like(x)
    for body in np.unique(owner):
        select = owner == body
        direct[select] = transport_material_nodes(
            x[select], opening[body], target[body]
        )
    anchors = read(bed)["attachment_rows"]
    ar = np.array([r["volume_row"] for r in anchors])
    if len(np.unique(ar)) != len(ar) or not np.array_equal(
        owner[ar], [r["donor"] for r in anchors]
    ):
        raise ValueError("Attachment identities differ")
    if (
        np.linalg.norm(x[ar] - np.array([r["posed_m"] for r in anchors]), axis=1).max()
        > 1.0e-6
    ):
        raise ValueError("Attachment coordinates differ from physical registration")
    outer = np.load(layer)["Orows"]
    # Volume-binned averages supply soft registration observations of donor
    # motion. Only native mechanical attachments are interpolation constraints.
    gridsize = 0.06
    key = np.floor((x - x.min(axis=0)) / gridsize).astype(int)
    _, bins = np.unique(key, axis=0, return_inverse=True)
    count = np.bincount(bins)
    d = direct - x
    aggregate = lambda a: np.column_stack(
        [np.bincount(bins, weights=v) / count for v in a.T]
    )
    centers = aggregate(x)
    shift = aggregate(d)
    # A cubic polyharmonic spline is twice continuously differentiable,
    # including across the original donor partitions. The affine tail gives
    # exact rigid-motion reproduction. Scaling lengths makes the regularizer dimensionless.
    h = gridsize
    o = x.mean(axis=0)
    centers = np.vstack((centers, x[ar]))
    shift = np.vstack((shift, d[ar]))
    reg = np.r_[np.ones(len(count)), np.zeros(len(ar))]
    spline = RBFInterpolator(
        (centers - o) / h, shift, kernel="cubic", degree=1, smoothing=reg
    )
    posed = x.copy()
    for i in range(0, len(x), 2048):
        posed[i : i + 2048] += spline((x[i : i + 2048] - o) / h)
    rounded = posed.astype("<f4").astype(float)
    err = np.linalg.norm(rounded[ar] - direct[ar], axis=1)
    if err.max() > 1.5e-7:
        raise ValueError(f"Lost exact attachment constraint {err.max()}")
    result = dict(
        source=pins,
        opening=sha(first),
        target=sha(endfile),
        ancestry=sha(cm),
        attachment_registration=sha(bed),
        grid_spacing_m=gridsize,
        soft_sites=len(count),
        hard_attachment_count=len(ar),
        max_attachment_error_m=float(err.max()),
        rms_distance_to_single_donor_m=float(
            np.sqrt(np.mean(np.sum((rounded - direct) ** 2, axis=1)))
        ),
        bed=dict(
            outer_nodes=len(outer),
            below=int(np.sum(rounded[outer, 2] < 0)),
            minimum_z_m=float(rounded[outer, 2].min()),
        ),
        cells=validate(ref, rounded, tets, role),
        scope="Offline Float32 material placement, preserving connectivity, source coordinates and mechanical attachment identities. No extra simulation forces or material mass. Contact and equilibrium require separate native verification.",
    )
    np.save(out / "positions.npy", rounded.astype("<f4"))
    result["candidate_sha256"] = sha(out / "positions.npy")
    (out / "quality.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    prep(*(Path(s).resolve() for s in sys.argv[1:]))
