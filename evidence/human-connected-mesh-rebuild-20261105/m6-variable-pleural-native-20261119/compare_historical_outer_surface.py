"""Measure registration and row identity of the archived zero-crossing outer skin."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

from fit_cardiac_volume_grid import HERE, PREVIOUS, hash_file
sys.path.insert(0,str(PREVIOUS))
from fit_skin_source import BASE, capture


def compare():
    asset = HERE/"historical-outer-zero-609.npz"
    with np.load(asset) as archive:
        points = archive["points_world_m"]
        original = archive["reference_original_vertex_rows"]
        labels = archive["reference_subdivision_parent_face_rows"]
        faces = archive["faces"]
    result=[]
    for path in (BASE, HERE/"cardiac-split-native-8000/native/accepted-geometry/step-8000.mrvpack"):
        receipt, xyz, triangles = capture(path)
        retained=np.unique(triangles)
        forward, _ = cKDTree(xyz[retained]).query(points)
        backward, _ = cKDTree(points).query(xyz[retained])
        direct = np.linalg.norm(points[original>=0] - xyz[original[original>=0]],axis=1)
        quantile=lambda x:(np.quantile(x,(0,.1,.5,.9,.99,1))*1000).tolist()
        row=dict(step=receipt["accepted_step"], pack_sha256=receipt["pack_file_sha256"],
                 matching_topology=faces.shape==triangles.shape,
                 historical_vertices=len(points), current_vertices=len(xyz),
                 historical_faces=len(faces), current_faces=len(triangles),
                 historical_to_current_nearest_mm=quantile(forward),
                 current_to_historical_nearest_mm=quantile(backward),
                 shared_row_hypothesis_residual_mm=quantile(direct))
        result.append(row)
        print(row,flush=True)
    (HERE/"historical-outer-registration.json").write_text(json.dumps(dict(
        asset_sha256=hash_file(asset), candidate="609", comparisons=result,
        note="Nearest-vertex distances characterize mismatch without establishing binding or surface clearance."),
        indent=2)+"\n")


if __name__=="__main__":
    compare()
