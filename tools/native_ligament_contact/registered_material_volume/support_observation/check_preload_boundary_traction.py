"""Account for where the quasistatic preload sends its elastic force across the movable boundary."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def laws(commands):
    compile = json.loads(commands.read_bytes())[0]
    result = {}
    for label, key in [
        (15, "NUMI_HUMAN_SKIN_DERMIS_MATERIAL"),
        (16, "NUMI_HUMAN_SUBCUTANEOUS_ADIPOSE_MATERIAL"),
    ]:
        (arg,) = [a for a in compile if a.startswith("-D" + key + "=")]
        path = Path(arg.split("=", 1)[1].strip('"'))
        text = path.read_text()
        if (
            "0.5 * shear * (I1() - 3 - 2 * log(J()))" not in text
            or "0.5 * bulk * pow(log(J()), 2)" not in text
        ):
            raise ValueError("Unrecognized energy for " + str(path))
        result[label] = [
            float(
                re.search(
                    r"parameter " + name + r"\s*:\s*MPa\s*=\s*([0-9.eE+-]+)", text
                )[1]
            )
            * 1e6
            for name in ("shear", "bulk")
        ]
    return result


def displacement(patch, count):
    nodes, delta = patch["node_rows"], patch["delta_m"]
    if (
        nodes.ndim != 1
        or nodes.dtype.kind not in "iu"
        or np.any(nodes < 0)
        or np.any(nodes >= count)
        or len(np.unique(nodes)) != len(nodes)
        or delta.shape != (len(nodes), 3)
        or not np.isfinite(delta).all()
    ):
        raise ValueError("Invalid nodal displacement proposal")
    return nodes, delta


def run(case, work, predictor, addition, out):
    cfg = json.loads((case / "launch.json").read_bytes())
    env = cfg["environment"]
    base = json.loads((predictor / "equilibrium.json").read_bytes())
    q = np.load(predictor / "proposal.npz")
    if sha(predictor / "proposal.npz") != base["proposal_sha256"]:
        raise ValueError("Wrong starting material")
    pfx = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
    rf = Path(env[pfx + "NODE"])
    ele = Path(env[pfx + "ELEMENTS"])
    cur = Path(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    for f in (rf, ele, cur):
        if sha(f) != cfg["pins"][str(f)]:
            raise ValueError("Different material source")
    x0 = np.loadtxt(cur, skiprows=1, usecols=(1, 2, 3))
    x = x0.copy()
    nodes, delta = displacement(q, len(x))
    x[nodes] += delta
    delta_source = base["proposal_sha256"]
    if base["reference_sha256"] != sha(rf) or base["current_sha256"] != sha(cur):
        raise ValueError("Preload was built on different tissue coordinates")
    if addition is not None:
        z = np.load(addition)
        idx, delta = displacement(z, len(x))
        if not np.isin(idx, nodes).all():
            raise ValueError("Correction leaves the audited material domain")
        x[idx] = x0[idx] + delta
        delta_source = sha(addition)
    rest = np.loadtxt(rf, skiprows=1, usecols=(1, 2, 3)).astype("<f4").astype(float)
    t = np.loadtxt(ele, skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=int)
    tags = t[:, 4]
    t = t[:, :4] - 1
    cavity = Path(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    if sha(cavity) != cfg["pins"][str(cavity)]:
        raise ValueError("Changed material topology")
    patch = np.load(cavity)
    kept = patch["retained_input_cell_rows"]
    new = patch["replacement_tetrahedra"]
    if set(tags[np.setdiff1d(np.arange(len(t)), kept)]) != {15}:
        raise ValueError("Expected the two registered dermal cavities")
    t = np.vstack((t[kept], new))
    tags = np.r_[tags[kept], np.full(len(new), 15)]
    hit = np.isin(t, nodes).any(1)
    t = t[hit]
    tags = tags[hit]
    M = laws(Path(cfg["argv"][0]).parent / "commands.json")
    n = np.load(work / "surface.npz")["normal"]
    H = (rest[t[:, 1:]] - rest[t[:, :1]]).transpose(0, 2, 1)
    V = np.linalg.det(H) / 6
    if not (
        np.isfinite(V).all()
        and (V > 0).all()
        and n.shape == (3,)
        and abs(np.linalg.norm(n) - 1) < 1e-7
    ):
        raise ValueError("Invalid stress-free reference or bed normal")
    B = np.linalg.inv(H)
    G = np.concatenate((-B.sum(axis=1, keepdims=True), B), axis=1)
    mu, K = np.asarray([M[l] for l in tags]).T

    def reaction(y):
        F = (y[t[:, 1:]] - y[t[:, :1]]).transpose(0, 2, 1) @ B
        J = np.linalg.det(F)
        if np.min(J) <= 0:
            raise ValueError("Inverted material")
        P = mu[:, None, None] * F + (K * np.log(J) - mu)[:, None, None] * np.linalg.inv(
            F
        ).transpose(0, 2, 1)
        force = -V[:, None] * (G @ (n @ P)[..., None])[..., 0]
        return np.bincount(t.ravel(), weights=force.ravel(), minlength=len(x))

    full = reaction(x) - reaction(x0)
    body = sum(full)
    scale = float(np.sum(abs(full)))
    if abs(body) > 1e-8 * max(scale, 1):
        raise ValueError("Unbalanced internal force")
    header = Path(cfg["argv"][0]).parent
    s = (
        (header / "NumiHumanLoadedSkinPoseData.inc")
        .read_text()
        .split("Anchor anchors[]={")[1]
        .split("};")[0]
    )
    anchors = np.array(re.findall(r"\{\d+u,(\d+)u,\d+u,\{", s), int)
    s = (
        (header / "NumiHumanCorticalSkinBinding.inc")
        .read_text()
        .split("inline constexpr Sample samples[]={")[1]
        .split("};")[0]
    )
    samples = np.array(re.findall(r"^\{(\d+),", s, re.MULTILINE), int)
    wall = np.unique(np.load(rf.parent / "material-walls.npz")["C"])
    types = np.full(len(x), 4)
    types[wall] = 3
    types[anchors] = 2
    types[samples] = 1
    types[nodes] = 0
    stats = {}
    for k, name in enumerate(
        [
            "movable_material",
            "cortical_binding_samples",
            "body_anchors",
            "other_cortical_boundary",
            "other_held_material",
        ]
    ):
        part = full[types == k]
        stats[name] = {
            "nodes_with_force": int(sum(abs(part) > 1e-10)),
            "net_material_force_along_bed_normal_N": float(sum(part)),
            "positive_N": float(np.maximum(part, 0).sum()),
            "negative_N": float(np.minimum(part, 0).sum()),
        }
    src = Path(env["NUMI_HUMAN_GLUTEAL_MATRIX"])
    ports = np.load(
        Path(env["NUMI_HUMAN_GLUTEAL_FAT_MASS_PREFLIGHT"]).parent
        / "material/parents.npz"
    )
    if sha(src) != cfg["pins"][str(src)]:
        raise ValueError("Skeletal force operator differs from the native release")
    gp = ports["fat"]
    nf = len(gp)
    nb = len(ports["donors"])
    N = 3 * nf + 6 * nb
    Kp = np.fromfile(src, "<f4").astype(float).reshape(N, N)
    disp = np.r_[(x[gp] - x0[gp]).ravel(), np.zeros(6 * nb)]
    fg = -(Kp @ disp)[3 * nf :].reshape(nb, 6)[:, :3]
    screen = {
        "candidate_sha256": delta_source,
        "normal": n.tolist(),
        "internal_force_closure_N": float(body),
        "elastic_force_change": stats,
        "gluteal_donor_force_change_normal_N": (fg @ n).tolist(),
        "sum_gluteal_donor_force_change_normal_N": float(sum(fg @ n)),
        "scope": "Incremental registered continuum stress relative to the opening geometry. Movable, explicit cortical, anatomical anchor, other cortical boundary and remaining fixed material rows are disjoint. Remaining fixed material must transmit this traction through the larger body. Holding those nodes in a reduced initialization does not establish a skeletal reaction.",
    }
    offset, count = map(
        int,
        re.search(
            r"registered_material_span object=15 element_offset=\d+ element_count=\d+ state_offset=(\d+) state_count=(\d+)",
            (case / "stderr.log").read_text(),
        ).groups(),
    )
    if count != len(x0):
        raise ValueError("Different native material row count")
    contact = np.fromfile(case / "contact-1.rows", "<f4").reshape(-1, 7, 4)
    ids = contact[:, 0, :].copy().view("<u4")[:, 1].astype(np.int64) - offset
    present = (ids >= 0) & (ids < count)
    observed = np.unique(ids[present])
    held = np.flatnonzero((types == 4) & (abs(full) > 1e-10))
    here = held[np.isin(held, observed)]
    screen["registered_release_row_comparison"] = {
        "total_traction_rows": len(held),
        "same_rows_in_native_generic_rigid_contact": len(here),
        "signed_change_on_same_rows_N": float(full[here].sum()),
        "scope": "Identity comparison with the actual generic rigid contacts of the terminal native microstep. Further neighbors and new contacts under preload are unmeasured.",
    }
    # Rank spatially connected pieces of the cutoff without arbitrary coordinate bins.
    import itertools

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    l = np.full(len(x), -1, int)
    l[held] = np.arange(len(held))
    edges = np.vstack([l[t[:, [i, j]]] for i, j in itertools.combinations(range(4), 2)])
    edges = edges[(edges >= 0).all(1)]
    cnt, labels = connected_components(
        coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(len(held), len(held)),
        ).tocsr(),
        directed=False,
    )
    groups = []
    for label in range(cnt):
        ids = held[labels == label]
        force = full[ids]
        pos = float(np.maximum(force, 0).sum())
        neg = float(np.minimum(force, 0).sum())
        groups.append(
            {
                "count": len(ids),
                "net_N": float(force.sum()),
                "opposing_magnitudes_N": [pos, neg],
                "positive_force_center_m": (
                    np.average(x0[ids], axis=0, weights=np.maximum(force, 0)).tolist()
                    if pos
                    else None
                ),
                "node_with_largest_force": int(ids[np.argmax(abs(force))]),
            }
        )
    screen["cut_boundary_components"] = sorted(
        groups, key=lambda q: abs(q["net_N"]), reverse=True
    )
    screen["peak_force_change_on_other_held_material_node_N"] = float(
        np.max(abs(full[types == 4]), initial=0.0)
    )
    out.write_text(json.dumps(screen, indent=2, allow_nan=False) + "\n")
    return screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_run", type=Path)
    parser.add_argument(
        "registered_bed_directory", type=Path, help="contains the matching surface.npz"
    )
    parser.add_argument(
        "preload_directory",
        type=Path,
        help="contains equilibrium.json and proposal.npz",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--corrected-proposal",
        type=Path,
        help="replacement displacements in the same material coordinates",
    )
    parser.add_argument(
        "--max-held-force-change-n",
        type=float,
        help="return failure if any other fixed material node needs a greater change of bed-normal force",
    )
    args = parser.parse_args()
    tol = args.max_held_force_change_n
    if tol is not None and (not np.isfinite(tol) or tol < 0):
        parser.error("Force tolerance must be finite and nonnegative")
    r = run(
        args.native_run,
        args.registered_bed_directory,
        args.preload_directory,
        args.corrected_proposal,
        args.output,
    )
    held = r["elastic_force_change"]["other_held_material"]
    print(
        f"Other fixed material receives {held['net_material_force_along_bed_normal_N']:.6g} N of net incremental traction across {held['nodes_with_force']} nodes."
    )
    print(
        f"The gluteal skeletal donors receive {r['sum_gluteal_donor_force_change_normal_N']:.6g} N."
    )
    if tol is not None and r["peak_force_change_on_other_held_material_node_N"] > tol:
        raise SystemExit("The regional preload requires unaccounted boundary traction.")


if __name__ == "__main__":
    main()
