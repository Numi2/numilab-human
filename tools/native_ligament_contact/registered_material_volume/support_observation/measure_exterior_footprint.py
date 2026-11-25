"""Measure the bed-plane projection of the accepted exterior within a specified gap."""

import argparse
import json
from pathlib import Path

from manifold3d import CrossSection
import numpy as np

from assess_supported_release import digest, read_momentum


def projected_area(points, faces, normal, offset, gap):
    """Union clipped triangles so overlapping tissue projections count only once."""
    normal = np.asarray(normal, dtype=float)
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
            or np.any((faces < 0) | (faces >= len(points)))):
        raise ValueError("Invalid exterior surface")
    if (normal.shape != (3,) or not np.isclose(np.linalg.norm(normal), 1, rtol=0, atol=1e-7)
            or not np.isfinite([offset, gap]).all() or gap < 0):
        raise ValueError("Expected a unit bed normal and a nonnegative gap")
    tangent = np.eye(3)[np.argmin(abs(normal))]
    tangent -= (tangent @ normal) * normal
    tangent /= np.linalg.norm(tangent)
    basis = np.column_stack((tangent, np.cross(normal, tangent)))
    projected = points @ basis
    height = points @ normal - offset - gap
    contours = []
    for face in faces[np.min(height[faces], axis=1) <= 0]:
        poly = []
        for i in range(3):
            a, b = face[i-1], face[i]
            if (height[a] <= 0) != (height[b] <= 0):
                fraction = height[a] / (height[a] - height[b])
                poly.append(projected[a] + fraction * (projected[b] - projected[a]))
            if height[b] <= 0:
                poly.append(projected[b])
        if len(poly) < 3:
            continue
        # Express planar coordinates in mm for polygon clipping. The caller uses SI.
        polygon = np.asarray(poly) * 1000
        x, y = (polygon - polygon[0]).T
        signed = x @ np.roll(y, -1) - y @ np.roll(x, -1)
        if signed:
            contours.append(polygon if signed > 0 else polygon[::-1].copy())
    return CrossSection(contours).area() * 1e-6 if contours else 0.0


def measure(run, cortex_report):
    cfg = json.loads((run/"launch.json").read_bytes())
    obs = json.loads(cortex_report.read_bytes())["accepted_observation"]
    nodes = run / "physical-accepted.node"
    receipt = run / "native-traces/accepted-geometry" / f"step-{obs['step']}.receipt.json"
    if (obs["launch_sha256"] != digest(run/"launch.json") or obs["accepted_node_sha256"] != digest(nodes)
            or obs["receipt_sha256"] != digest(receipt)):
        raise ValueError("Mismatched accepted material capture")
    walls = Path(cfg["environment"]["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"]).with_name("material-walls.npz")
    reference = json.loads((cortex_report.parent/"all-wall-cortex.json").read_bytes())
    if (digest(walls) != reference["wall_sha256"] or reference["launch_sha256"] != obs["launch_sha256"]
            or reference["receipt_sha256"] != obs["receipt_sha256"]):
        raise ValueError("Physical tissue wall hash does not match the complete crossing audit")
    x = np.loadtxt(nodes, skiprows=1, usecols=(1, 2, 3))
    with np.load(walls) as z:
        faces = z["O"]
    with (run/"stdout.log").open() as stream:
        source, = (r for r in stream if r.startswith("resting_fem_skin_bed_source=enabled "))
    fields = dict(t.split("=", 1) for t in source.split() if "=" in t)
    n = np.fromstring(fields["normal"], sep=",")
    if fields["proxy_count"] != "1" or not np.isclose(np.linalg.norm(n), 1, rtol=0, atol=1e-7):
        raise ValueError("Expected one fixed registered support plane")
    step, _, _, _, gravity, _, normal, _ = read_momentum(run)
    if step != obs["step"] or not np.allclose(n, normal, atol=1e-6, rtol=0):
        raise ValueError("Bed and accepted momentum normals differ")
    weight = -float(gravity @ normal)
    results = []
    for mm in (0,1,2,5,10,20,40):
        area = projected_area(x, faces, n, float(fields["plane_offset_m"]), mm / 1000)
        results.append(dict(gap_mm=mm,area_cm2=area*1e4,
                            mean_pressure_for_skeletal_weight_kPa=weight/(area*1000) if area else None))
    return dict(accepted_s=obs["time_s"],receipt=obs["receipt_sha256"],walls_sha256=digest(walls),
                skeletal_gravity_normal_N=weight, terminal_bed_N=obs["bed_support"]["last_root_mean_normal_force_N"],
                fixed_geometry=results,scope="Projected area of the observed exterior below each plane offset. The area measures geometric availability before any deformation. The pressure is a necessary mean at that area to support the skeleton alone.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","cortex_report","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(measure(args.run,args.cortex_report),indent=2,allow_nan=False)+"\n")
