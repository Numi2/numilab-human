"""Measure the actual area admitted by the fixed-bed contact law at the fitted pose."""
import json
from pathlib import Path
import sys
import numpy as np
from compare_bed_sampling import clipped_simplices, integrate
from material_context import sha


def measure(root):
    source = root/"load/support.npz"
    if sha(source) != json.loads((root/"load/input.json").read_text())["data"]:
        raise ValueError("Support geometry identity differs")
    native = json.loads((root/"native-wrench.json").read_text())["rows"]["emitted"]
    with np.load(source) as z:
        d = dict(z)
    with np.load(root/"wrench-fit/preload.npz") as fit:
        y = d["y"].copy()
        y[d["free"]] = fit["coordinates"]
    w = float(d["width"])
    if d["excluded_min"] <= w + np.abs(y[:,2]-d["y"][:,2]).max():
        raise ValueError("Incomplete contact surface")
    faces = d["f"]
    ids, bary = clipped_simplices(y[faces,2], w)
    triangles = bary @ y[faces[ids]]
    cross = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    projected = abs(cross[:,2]).sum() * .5
    area = np.linalg.norm(cross,axis=1).sum() * .5
    f = integrate(y, faces, d["area"], d["density"], float(d["dt"]), w, 4)[0]
    if projected <= 0 or projected > area*(1+1.e-12) or not np.isclose(f.sum(),native["normal_N"],rtol=1.e-6):
        raise ValueError("Invalid admitted area or mismatch with native resultant")
    cumulative = np.cumsum(np.sort(f)[::-1]) / f.sum()
    result = dict(fitted_pose_sha256=sha(root/"wrench-fit/preload.npz"),
                  width_m=w, contributing_faces=len(np.unique(ids)), positive_load_nodes=int((f>0).sum()),
                  sum_projected_area_cm2=float(projected*1.e4), surface_area_cm2=float(area*1.e4),
                  native_bed_force_N=native["normal_N"], maximum_nodal_force_N=float(f.max()),
                  nodes_for_half_total_load=int(np.searchsorted(cumulative,.5)+1),
                  nominal_normal_stress_kPa=float(native["normal_N"]/projected/1.e3),
                  scope="Area is the sum of clipped triangles projected onto the bed. Overlapping projections "
                  "would reduce the union. The reported stress is force divided by this sum, not a measured pressure map.")
    (root/"bed-certificate/support-distribution.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    print(measure(Path(sys.argv[1]).resolve()))
