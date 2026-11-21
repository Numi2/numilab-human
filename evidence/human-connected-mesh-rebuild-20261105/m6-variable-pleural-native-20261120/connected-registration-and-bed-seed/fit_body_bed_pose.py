"""Compute a geometric bed support pose from the retained registered outer skin."""
import json
import numpy as np
from native_source import run, skin
import csv
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

native = run/'native'
sample = next(csv.DictReader((native/'resting-com-momentum-diagnostic.csv').open()))
proxy = np.asarray([float(sample[f'com_{axis}_m']) for axis in 'xyz'])
mass = float(sample['total_body_mass_kg'])
launch = json.loads((run/'launch.json').read_text())['argv']
offset = launch.index('--root-pose')+1
q = np.array(launch[offset:offset+7],float)
hull = ConvexHull(skin)

def contact_metrics(points, g):
    gap = points[:,2]-points[:,2].min()
    metrics = {}
    for width in (1e-5,1e-3):
        rows = np.flatnonzero(gap<=width)
        projected = points[rows,:2]
        # Degenerate support needs a finite margin for passive equilibrium.
        try:
            poly = ConvexHull(projected)
            margin = float(-(poly.equations[:,:2]@g[:2]+poly.equations[:,2]).max())
            footprint = float(poly.volume)
        except QhullError:
            margin = None
            footprint = 0.
        metrics[str(width)] = dict(nodes=len(rows),footprint_m2=footprint,
                                   com_projected_margin_m=margin,
                                   longitudinal_span_m=float(np.ptp(projected[:,1])))
    return metrics

baseline = contact_metrics(skin.astype(float),proxy)
target = np.array([0.,0.,1.])
best = None
# Every triangle of the lower convex hull defines a support plane with
# nonnegative gap. Select a geometric seed near the authored pose which can
# bear a vertical point load through the measured articulated COM.
for number, (face, plane) in enumerate(zip(hull.simplices,hull.equations)):
    up = -plane[:3]
    tilt = np.arccos(np.clip(up@target,-1,1))
    if tilt>np.deg2rad(5):
        continue
    projected = proxy - ((proxy-skin[face[0]])@up)*up
    corner = skin[face].astype(float)
    d = corner[1:]-corner[0]
    bary = np.linalg.lstsq(d.T,projected-corner[0],rcond=None)[0]
    if bary.min() < 0 or bary.sum() > 1:
        continue
    if best is None or tilt < best[0]:
        best = (tilt,number,up)
assert best is not None
tilt, number, up = best
if tilt:
    direction = np.cross(up,target)
    rotate = Rotation.from_rotvec(tilt*direction/np.linalg.norm(direction))
else:
    rotate = Rotation.identity()
posed = rotate.apply(skin-q[:3])+q[:3]
center = rotate.apply(proxy-q[:3])+q[:3]
vertical_drop = float(posed[:,2].min())
posed[:,2]-=vertical_drop
center[2]-=vertical_drop
qd = q.copy()
qd[:3][2]-=vertical_drop
qd[3:] = (rotate*Rotation.from_quat(q[3:])).as_quat()
print(json.dumps(dict(original_root_pose=q.tolist(),trial_root_pose=qd.tolist(),
                      tilt_deg=float(np.rad2deg(tilt)),rigid_COM_proxy_kg=mass,
                      minimum_gap_before_m=float(skin[:,2].min()),lower_hull_triangle=skin[hull.simplices[number]].tolist(),
                      original=baseline,trial=contact_metrics(posed,center),
                      scope="rigid alignment search at authored geometry; uses step-1 articulated COM as a geometric proxy and preserves all relative positions; no pressure, deformation or settled-state claim"),indent=2))
