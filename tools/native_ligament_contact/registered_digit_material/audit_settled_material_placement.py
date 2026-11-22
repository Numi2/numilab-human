"""Test whether the detailed material can inherit the settled control's donor transforms."""

import json, csv, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from fit_dorsal_material import inputs
from material_context import sha
from torso_cell_geometry import geom


def audit(compact, native, report, split, out):
    targetfile = compact / "terminal.receipt.json"
    state = json.loads(targetfile.read_text())
    if state["accepted_step"] != 30000 or state["physical_endpoint"] != "accepted":
        raise ValueError("Missing actual 60 s endpoint")
    # The same native receipt contains opening geometry. Explicitly select the accepted transform.
    end = {r["body_index"]: r for r in state["accepted_registered_body_poses"]}
    srcfile = native / "registered-initial-bodies.json"
    original = {
        r["body_index"]: r["current"] for r in json.loads(srcfile.read_text())["bodies"]
    }
    src, loaded, tets, role, ids = inputs(native)
    ref = src.astype(float)
    y = loaded.astype(float)
    measures = json.loads(report.read_text())
    launch = json.loads((native / "launch.json").read_text())
    if measures["fine"] != sha(native / "launch.json") or measures["compact"] != sha(
        compact / "launch.json"
    ):
        raise ValueError("Matched donor loads differ")
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    ancestry = Path(launch["environment"][prefix + "COMBINED_NODE_MAP_CSV"])
    if sha(ancestry) != measures["dense_map"]:
        raise ValueError("Different registered ancestry")
    owner = np.full(len(y), -1, dtype=int)
    with ancestry.open() as f:
        for rec in csv.DictReader(f):
            i = int(rec["joint_node_id"]) - 1
            d = int(rec["donor_body"])
            if owner[i] not in (-1, d):
                raise ValueError("Ambiguous bonded material donor")
            owner[i] = d
    if np.any(owner < 0):
        raise ValueError("Material node without donor")
    trial = np.empty_like(y)
    for d in np.unique(owner):
        a = original[d]
        b = end[d]
        prev = np.array(a["local_to_world_row_major_f32"]).reshape(3, 3)
        nxt = Rotation.from_quat(b["quaternion_xyzw"]).as_matrix()
        sel = owner == d
        local = np.linalg.solve(prev, (y[sel] - np.array(a["position_world_m"])).T)
        trial[sel] = (nxt @ local).T + b["position_m"]
    outer = np.load(split / "physical.npz")["Orows"]
    support = {}
    for b, info in measures["donors"].items():
        sel = outer[owner[outer] == int(b)]
        if not len(sel):
            continue
        support[b] = dict(
            compact_load_N=info["compact_mean_N"],
            original_lowest_m=float(y[sel, 2].min()),
            posed_lowest_m=float(trial[sel, 2].min()),
            posed_below_plane=int(np.sum(trial[sel, 2] < 0)),
        )
    below = int(np.sum(trial[outer, 2] < 0))
    failed = {}
    for typ in (15, 16):
        cells = tets[role == typ]
        ninv = nshape = nstrain = 0
        for i in range(0, len(cells), 32768):
            c = cells[i : i + 32768]
            v, q = geom(trial, c)
            vr, _ = geom(ref, c)
            J = v / vr
            ninv += int(np.sum(v <= 0))
            nshape += int(np.sum((v <= 1e-18) | (q < 1e-5)))
            nstrain += int(np.sum((J < 0.4) | (J > 2.5)))
        failed[typ] = dict(
            cells=len(cells),
            inverted=ninv,
            failed_shape_or_volume=nshape,
            outside_installed_volume_ratio=nstrain,
        )
    result = dict(
        native_material_source=ids,
        pose_source=sha(srcfile),
        terminal_pose=sha(targetfile),
        donor_report=sha(report),
        posed_outer_below_bed=below,
        rows=support,
        volume=failed,
        scope="Counterfactual rigid-donor transport of every installed material node to the actual 60 s registered skeleton. No forces applied and no trial geometry installed.",
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(below, failed)


if __name__ == "__main__":
    audit(*(Path(a).resolve() for a in sys.argv[1:]))
