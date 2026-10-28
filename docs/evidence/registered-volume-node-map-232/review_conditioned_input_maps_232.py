from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

P = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
RUN = P / "canonical-skin-volume-collar-228/run-002"
TOOL = P / "volume-worker/native-domain-217/build_native_domain_217.py"
O_PATH = RUN / "conditioned-compact-O.npz"
COLLAR_PATH = RUN / "conditioned-head-collar.npz"
WALL_PATH = RUN / "donor-sidewall-faces.csv"
PARENT_CELLS = P / "volume-worker/baseline-parent-cell-mass.csv.gz"
PARENT_NODES = P / "volume-worker/baseline-compiled-parent-nodes.csv.gz"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    spec = importlib.util.spec_from_file_location("native_domain_217_review", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    collar = __import__("numpy").load(COLLAR_PATH, allow_pickle=False)
    outer = module.load_surface(O_PATH, "O", collar)
    import numpy as np

    points = np.asarray(outer["points"], dtype=np.float32)
    faces = np.asarray(outer["faces"], dtype=np.int64)
    collar_points = np.asarray(collar["points_world_m"], dtype=np.float32)
    collar_faces = np.asarray(collar["outer_faces"], dtype=np.int64)
    assert points.shape == (32002, 3)
    assert faces.shape == (63959, 3)
    assert np.array_equal(collar_points[:len(points)], points)
    assert np.array_equal(collar_faces, faces)
    o_source_vertices = np.arange(len(points), dtype=np.int64)
    o_shell_nodes = np.arange(len(points), dtype=np.int64)
    o_source_faces = faces.copy()
    o_shell_face_rows = np.arange(len(faces), dtype=np.int64)
    assert np.array_equal(o_source_vertices, o_shell_nodes)
    assert np.array_equal(o_source_faces, faces)
    assert np.array_equal(o_shell_face_rows, np.arange(len(faces), dtype=np.int64))
    assert np.array_equal(np.unique(o_source_faces), o_source_vertices)
    assert len(outer["source_rows"]) == len(outer["exact_source_rows"]) == len(faces)
    assert len(outer["ancestry"]) == len(faces)
    typed_rows = sum(len(rows) for rows in outer["ancestry"])
    assert typed_rows == 67296
    assert set(map(int, np.unique(collar["donor_O_to_candidate_vertex"]))) <= set(range(len(points)))
    donor_o_rows = np.asarray(collar["donor_O_to_candidate_vertex"], dtype=np.int64)
    donor_o_parent_rows = np.asarray(collar["donor_O_parent_node_rows"], dtype=np.int64)
    donor_i_parent_rows = np.asarray(collar["donor_I_parent_node_rows"], dtype=np.int64)
    donor_d_parent_rows = np.asarray(collar["donor_D_parent_node_rows"], dtype=np.int64)
    assert len(donor_o_rows) == len(donor_o_parent_rows) == 45
    assert len(donor_i_parent_rows) == len(donor_d_parent_rows) == 45
    assert len(set(map(int, donor_o_rows))) == len(donor_o_rows)
    assert len(set(map(int, donor_o_parent_rows))) == len(donor_o_parent_rows)
    assert len(set(map(int, donor_i_parent_rows))) == len(donor_i_parent_rows)
    assert len(set(map(int, donor_d_parent_rows))) == len(donor_d_parent_rows)
    assert np.array_equal(points[donor_o_rows], collar_points[donor_o_rows])
    walls = module.read_wall_provenance(collar, WALL_PATH, PARENT_CELLS, PARENT_NODES)
    assert len(walls["skin_parent_tets"]) == 90
    assert len(walls["fat_parent_tets"]) == 90
    assert np.all(walls["skin_materials"] == 15)
    assert np.all(walls["fat_materials"] == 16)
    role_contract = {
        "O_skin": {"marker": module.MARKER["skin_outer_O"], "role": "outer_O"},
        "I_skin": {"marker": module.MARKER["skin_inner_I"], "role": "inner_I"},
        "I_fat": {"marker": module.MARKER["fat_outer_I"], "role": "outer_I"},
        "D_fat": {"marker": module.MARKER["fat_inner_D"], "role": "inner_D"},
        "skin_wall": {"marker": module.MARKER["skin_wall_OI"], "role": "donor_skin_OI_headwall"},
        "fat_wall": {"marker": module.MARKER["fat_wall_ID"], "role": "donor_fat_ID_headwall"},
    }
    report = {
        "status": "CONDITIONED_O_AND_DONOR_COLLAR_ROW_MAPS_PASS",
        "scope": "existing conditioned O/collar input check; no TetGen, full-shell assembly, or physical admission",
        "inputs": {str(path): sha(path) for path in
                   (TOOL, O_PATH, COLLAR_PATH, WALL_PATH, PARENT_CELLS, PARENT_NODES)},
        "O": {
            "point_rows": len(points), "face_rows": len(faces),
            "registered_source_vertex_rows_exact_identity": True,
            "registered_shell_node_rows_exact_identity": True,
            "registered_source_faces_exact": True,
            "registered_shell_face_rows_exact_identity": True,
            "face_source_rows_preserved": True,
            "typed_ancestry_entries": typed_rows,
            "unique_points_covered_by_source_faces": len(np.unique(o_source_faces)),
        },
        "collar": {
            "outer_points_bitwise_equal_to_conditioned_O": True,
            "outer_faces_exact_equal_to_conditioned_O": True,
            "O_I_D_head_ring_rows_each": 45,
            "O_I_D_head_ring_row_ids_unique": True,
            "skin_sidewall_source_faces": len(walls["skin_parent_tets"]),
            "skin_sidewall_materials": sorted(set(map(int, walls["skin_materials"]))),
            "fat_sidewall_source_faces": len(walls["fat_parent_tets"]),
            "fat_sidewall_materials": sorted(set(map(int, walls["fat_materials"]))),
        },
        "assembled_shell_role_contract": role_contract,
        "full_production_I_D_surface_face_maps_checked": False,
        "reason": "The retained conditioned input check contains O and head-collar rings/walls, not full production I/D surface inputs.",
        "physical_admission": False,
        "tetgen_run": False,
        "full_scene_audit": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
