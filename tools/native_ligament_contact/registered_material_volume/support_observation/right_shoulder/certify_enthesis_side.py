"""Verify that the intended scapular traction patch faces infraspinatus in both registered sources."""

import json
import sys
from pathlib import Path

import igl
import numpy as np
from assemble_material import frames
from audit_registered_skeleton import source_geometry
from compile_connected_joint import digest
from rebind_ligament_material import banks
from registration_stages import parts
from repair_registered_surface import state
from scipy.spatial.transform import Rotation


def test(w, case, fit, out):
    c = json.loads((case / "launch.json").read_bytes())
    a = c["argv"]
    r = json.loads((fit / "candidates.json").read_bytes())
    source = Path(a[a.index("--soft-tissue-payload") + 1])
    poses = frames(case)[-1]
    pose = poses[34]
    R = Rotation.from_quat(pose["quaternion_xyzw"])

    def local(x):
        return R.inv().apply(x - pose["position_m"])

    names = {}
    for i in [83, 85]:
        v, wt, b, f, aff = parts(source, i)
        names[str(i)] = local(state(v, wt, aff, b, poses)), f
    ref = json.loads((w / "source-support-column.json").read_bytes())
    q = ref["frame"]
    x, f = source_geometry((w / "atlas/FJ1500.obj").read_bytes())
    names["FJ1500"] = local(x @ q["linear_m_per_mm"] + q["translation_m"]), f
    f, v, _ = banks(Path(a[3]))[34, 10]
    # Query unsigned bone distances on both sides before assigning the positive-winding exterior.
    rows = []
    steps = np.array([0.05, 0.25, 0.5, 1, 2, 3, 5, 8]) * 1e-3
    for i, pr in enumerate(r["proposals"]):
        sites = np.array([pr["point"]] + pr["active_nodes_m"])
        dist, ids, centers = igl.point_mesh_squared_distance(sites, v, f.astype("i4"))
        normals = np.cross(v[f[ids, 1]] - v[f[ids, 0]], v[f[ids, 2]] - v[f[ids, 0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        inside = (
            abs(igl.winding_number(v, f.astype("i4"), centers + 0.00005 * normals))
            > 0.5
        )
        normals[inside] *= -1
        rays = centers[:, None] + normals[:, None] * steps[None, :, None]

        def winding(s):
            return (
                np.abs(
                    igl.winding_number(
                        np.ascontiguousarray(s[0]),
                        np.ascontiguousarray(s[1], dtype="i4"),
                        rays.reshape(-1, 3),
                    )
                ).reshape(len(sites), len(steps))
                > 0.5
            )

        occupied = {n: winding(s) for n, s in names.items()}
        bon = winding((v, f))
        probes = occupied["83"] & occupied["FJ1500"] & ~occupied["85"] & ~bon
        rows.append(
            dict(
                proposal=i,
                center_proven=bool(probes[0].any()),
                proven_source_sites=probes.any(1).tolist(),
                max_cortical_gap_m=float(np.sqrt(dist.max())),
                depths_m=steps.tolist(),
                common_infraspinatus=probes.tolist(),
                subscapularis=occupied["85"].tolist(),
            )
        )
    out.write_text(
        json.dumps(dict(source=digest(fit / "candidates.json"), rows=rows), indent=2)
        + "\n"
    )
    print([(r["proposal"], r["proven_source_sites"]) for r in rows], flush=True)


if __name__ == "__main__":
    test(*(Path(s).resolve() for s in sys.argv[1:]))
