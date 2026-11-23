"""Resolve the exact linear-blend displacement of registered material skin near each physical origin."""

import json, sys, runpy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from track_elbow_attachments import prepare, locate
from measure_forearm_enclosure import Enclosure
from reweight_forearm_skin import pose_rows


def diagnose(case, base, reader, dec, lib, routes):
    s, ids, f, frames, rr = prepare(case, base, reader, dec, lib, routes)
    api = runpy.run_path(str(dec))
    origins = {
        p["source_site"]: p
        for bank in rr["banks"]
        for p in bank["sites"]
        if p["role"] == "origin"
    }
    ref, v, _ = frames["neutral"]
    cur = frames["terminal"][0]
    donors = s["binds"]["body"]
    rows = []

    def world(site, pose):
        m = pose[site["body"]]
        return (
            R.from_quat(m["quaternion_xyzw"]).apply(site["local_m"]) + m["position_m"]
        )

    sh = Enclosure(lib, v, f)
    _, fa, wa = locate(sh, v, f, np.array([world(x, ref) for x in origins.values()]))
    sh.close()

    def points(pose, vertices):
        A, b = api["reference_transforms"](s, pose_rows(pose), [0.0, 0.0, 0.0])
        return np.einsum("bij,vj->vbi", A, s["verts"][vertices, :3]) + b

    for m, face, ww in zip(origins.values(), fa, wa):
        vertices = ids[f[face]]
        x = points(ref, vertices)
        z = points(cur, vertices)
        y0 = ref[m["body"]]
        y1 = cur[m["body"]]
        rot = (
            R.from_quat(y1["quaternion_xyzw"])
            * R.from_quat(y0["quaternion_xyzw"]).inv()
        )
        expect = (x - y0["position_m"]) @ rot.as_matrix().T + y1["position_m"]
        contrib = np.einsum("v,vb,vbd->bd", ww, s["weights"][vertices], z - expect)
        radial = ww @ v[f[face]] - world(m, ref)
        radial /= np.linalg.norm(radial)
        radial = rot.apply(radial)
        projected = contrib @ radial * 1000
        predicted = contrib.sum(axis=0)
        render = ww @ frames["terminal"][1][f[face]]
        check = render - (
            rot.apply(ww @ v[f[face]] - y0["position_m"]) + y1["position_m"]
        )
        rows.append(
            dict(
                muscle=m["muscle"],
                face=int(face),
                closure_residual_mm=float(1000 * np.linalg.norm(predicted - check)),
                actual_radial_relative_motion_mm=float(check @ radial * 1000),
                donors=[
                    dict(
                        body=int(donors[k]),
                        weight=float(ww @ s["weights"][vertices, k]),
                        radial_mm=float(projected[k]),
                    )
                    for k in np.argsort(projected)
                ],
            )
        )
    return rows


if __name__ == "__main__":
    print(json.dumps(diagnose(*(Path(k) for k in sys.argv[1:])), indent=2))
