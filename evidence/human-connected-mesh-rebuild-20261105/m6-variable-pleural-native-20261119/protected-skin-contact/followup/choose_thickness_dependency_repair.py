"""Find the minimum set of thickness anchors to recook for terminal crossings."""
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from fit_recooked_skin import NATIVE, SKIN, TABLE, dependencies
from fit_clearance_patches import clear, decode, mesh_edges
from run_support_dependency_preflight import HERE, digest, read, write

p, faces, _, _, pins, proof = dependencies(decode(SKIN.read_bytes()))
edges, counts = mesh_edges(faces)
unchanged = np.zeros(len(p), dtype=bool)
unchanged[edges[counts != 2].ravel()] = True
unchanged[clear._exact_coordinate_tied_source_vertex_ids(p)] = True
with (NATIVE/"resting-local-bed-sections.csv").open() as stream:
    anchors = np.array([int(row["source_geometry_index"]) for row in csv.DictReader(stream)])
unchanged[anchors] = True
unchanged[anchors-1] = True
assert not np.any(unchanged & ~pins)

run = HERE/"recomputed-support-native"
readback = read(run/"step-8000-skin-exact.json")
locked_pairs = [(int(s["semantic"]), int(s["stable_id"]), a, b)
                for s in readback["surfaces"] for a,b in s["triangle_pairs"] if pins[faces[a]].all()]
ineligible = sorted({a for _,_,a,_ in locked_pairs if unchanged[faces[a]].all()})
face_ids = sorted({a for _,_,a,_ in locked_pairs} - set(ineligible))
available = [row[~unchanged[row]] for row in faces[face_ids]]
vertices, inv = np.unique(np.concatenate(available), return_inverse=True)
ri = np.repeat(np.arange(len(available)), [len(row) for row in available])
matrix = coo_matrix((np.ones(len(ri)), (ri, inv)), shape=(len(available),len(vertices))).tocsc()
# Minimize the number of edited sample positions. The small secondary objective
# makes the order repeatable without outweighing a single additional vertex.
objective = 1.0 + np.arange(len(vertices), dtype=float) / (1.0 + len(vertices)**2)
solution = milp(objective, integrality=np.ones(len(vertices),dtype=int),
                bounds=Bounds(0.,1.), constraints=LinearConstraint(matrix,1.,np.inf))
assert solution.success
selected = vertices[solution.x >= .5]
assert not np.any(unchanged[selected]) and pins[selected].all()
trial = pins.copy()
trial[selected] = False
assert not np.any(trial[faces[face_ids]].all(axis=1))
write(HERE/"thickness-dependency-repair.json", {
    "scene_pack_sha256": readback["accepted_pack_sha256"],
    "source_skin_sha256": digest(SKIN),
    "thickness_table_sha256": digest(TABLE),
    "intersection_witnesses_with_three_pinned_vertices": len(locked_pairs),
    "distinct_fully_pinned_faces": len(face_ids)+len(ineligible),
    "heavily_constrained_faces_without_any_movable_thickness_vertex": ineligible,
    "intersection_witnesses_unlocked_by_proposal": sum(not trial[faces[a]].all() for _,_,a,_ in locked_pairs),
    "minimum_number_of_thickness_positions_to_modify": len(selected),
    "selected_thickness_positions": list(map(int,selected)),
    "scope": "geometric degrees of freedom only, no fitted repair or contact changes",
})
