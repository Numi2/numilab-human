"""Find the minimum set of thickness anchors to recook for terminal crossings."""
import csv
from collections import Counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from fit_recooked_skin import NATIVE, SKIN, TABLE, dependencies
from fit_clearance_patches import clear, decode, mesh_edges, lattice
from classify_native_crossing_geometry import audit_integer, layer
from fit_thickness_dependency_repair import capture_in_memory
from register_tissue_bellies import READER
from numilab_human import cardiac_cavity_intersections as geometry
from run_support_dependency_preflight import HERE, digest, read, write

p, faces, _, _, pins, proof = dependencies(decode(SKIN.read_bytes()))
edges, counts = mesh_edges(faces)
unchanged = np.zeros(len(p), dtype=bool)
unchanged[edges[counts != 2].ravel()] = True
unchanged[clear._exact_coordinate_tied_source_vertex_ids(p)] = True
with (NATIVE/"resting-local-bed-sections.csv").open() as stream:
    anchors = np.array([int(row["source_geometry_index"]) for row in csv.DictReader(stream)])
# The native cook subtracts one from the persisted source identity.
authority = HERE/"recomputed-support-build/NumiHumanRestingVisual.hpp"
assert "const unsigned seed=row.sourceGeometryIndex-1u;" in authority.read_text()
unchanged[anchors-1] = True
assert not np.any(unchanged & ~pins)

run = HERE/"recomputed-support-native"
readback = read(run/"step-8000-skin-exact.json")
locked_pairs = [(int(s["semantic"]), int(s["stable_id"]), a, b)
                for s in readback["surfaces"] for a,b in s["triangle_pairs"] if pins[faces[a]].all()]
constraints = {}
tables = {}
for semantic, source, a, b in locked_pairs:
    tables.setdefault((semantic,source), []).append((a,b))
pack = run/"native/accepted-geometry/step-8000.mrvpack"

def coincidence_support(point, tri):
    vertices = [j for j, corner in enumerate(tri) if tuple(point)==tuple(corner)]
    if vertices:
        return vertices
    for left,right in ((0,1),(1,2),(2,0)):
        edge = geometry._sub(tri[right],tri[left])
        rel = geometry._sub(point,tri[left])
        if not any(geometry._cross(edge, rel)) and 0<=geometry._dot(edge,rel)<=geometry._dot(edge,edge):
            return [left,right]
    return [0,1,2]

with capture_in_memory(pack) as (mapping,bodies,receipt):
    assert receipt["pack_file_sha256"]==readback["accepted_pack_sha256"]
    pts, emitted = next(READER["primitives"](mapping,bodies,skin_only=True))[2:]
    assert np.array_equal(emitted,np.searchsorted(np.unique(faces),faces))
    left = layer(pts,emitted,sorted({a for _,_,a,_ in locked_pairs}))
    for key,_,positions,target_faces in READER["primitives"](mapping,bodies):
        if key not in tables:
            continue
        witnesses=tables.pop(key)
        target=layer(positions,target_faces,sorted({b for _,b in witnesses}))
        for a,b in witnesses:
            xs=audit_integer(left[a],target[b])
            assert xs
            rows = {tuple(sorted(faces[a,j] for j in coincidence_support(x,left[a]))) for x in xs}
            constraints[(*key,a,b)]=[tuple(i for i in row if not unchanged[i]) for row in sorted(rows)]
    assert not tables
impossible = sorted(key for key,c in constraints.items() if any(not row for row in c))
rows = sorted({row for group in constraints.values() for row in group if row})
vertices, inv = np.unique(np.concatenate(rows),return_inverse=True)
ri = np.repeat(np.arange(len(rows)),[len(row) for row in rows])
matrix = coo_matrix((np.ones(len(ri)), (ri, inv)),shape=(len(rows),len(vertices))).tocsc()
objective = 1.0 + np.arange(len(vertices),dtype=float)/(1.0+len(vertices)**2)
solution = milp(objective, integrality=np.ones(len(vertices),dtype=int),
                bounds=Bounds(0.,1.),constraints=LinearConstraint(matrix,1.,np.inf))
assert solution.success
selected = vertices[solution.x>=.5]
staged = set(map(int,selected))
assert not np.any(unchanged[selected]) and all(staged.intersection(row) for row in rows)
previous = set(read(HERE/"thickness-dependency-repair.json")["selected_thickness_positions"])
lost = sorted(key for key,group in constraints.items() if any(not previous.intersection(row) for row in group))
write(HERE/"native-index-edge-dependency-repair.json", {
    "native_weighted_seed_source_sha256":digest(authority),
    "scene_pack_sha256": readback["accepted_pack_sha256"],
    "exact_crossing_pairs": len(constraints),
    "necessary_constraints_by_movable_vertex_count": dict(Counter(map(len,rows))),
    "pairs_with_boundary_or_contact_anchors_needed": impossible,
    "pairs_still_forced_by_pinned_edges_with_initial_face_plan": len(lost),
    "minimum_thickness_positions_to_address_remaining_constraints": len(selected),
    "selected_thickness_positions": sorted(staged),
    "scope":"Exact existing intersections create necessary constraints. A feasible pin selection is not collision clearance."
})
