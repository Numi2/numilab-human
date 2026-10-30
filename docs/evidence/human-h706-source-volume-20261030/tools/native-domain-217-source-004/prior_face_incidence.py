import numpy as np
from typing import Any
def prior_face_incidence(mesh: dict[str, Any]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    points = np.asarray(mesh["points"], dtype=np.float32).astype(np.float64)
    boundary: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    local_faces = ((0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0))
    for cell_row, tet in enumerate(np.asarray(mesh["tetrahedra"], dtype=np.int64)):
        for a, b, c, opposite in local_faces:
            face = [int(tet[a]), int(tet[b]), int(tet[c])]
            pa, pb, pc = points[face]
            po = points[int(tet[opposite])]
            if float(np.dot(np.cross(pb - pa, pc - pa), po - pa)) > 0.0:
                face[1], face[2] = face[2], face[1]
            key = tuple(sorted(face))
            boundary.setdefault(key, []).append({
                "element_row": cell_row,
                "element_id": int(mesh["element_ids"][cell_row]),
                "oriented_nodes": tuple(face),
                "material_index": int(mesh["material_indices"][cell_row]),
            })
    for key, rows in boundary.items():
        if len(rows) > 2:
            raise ValueError(f"TetGen mesh has a nonmanifold tetrahedron face {key}")
    return boundary

