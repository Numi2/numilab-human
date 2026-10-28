from __future__ import annotations

import csv
import gzip
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("build_native_domain_217.py")
spec = importlib.util.spec_from_file_location("native_domain_217", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def annulus(a: int, b: int) -> np.ndarray:
    faces = []
    for i in range(3):
        j = (i + 1) % 3
        faces.extend(((a + i, a + j, b + j), (a + i, b + j, b + i)))
    return np.asarray(faces, dtype=np.int64)


def save_fixture_npzs(root: Path) -> tuple[Path, Path, Path, Path]:
    outer_points = np.asarray([[-1, -1, 2], [1, -1, 2], [0, 1, 2]], dtype=np.float32)
    inner_points = np.asarray([[-.5, -.5, 1], [.5, -.5, 1], [0, .5, 1]], dtype=np.float32)
    deep_points = np.asarray([[-.4, -.4, 0], [.4, -.4, 0], [0, .4, 0]], dtype=np.float32)
    collar_points = np.vstack((outer_points, inner_points, deep_points))
    outer_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    skin_wall = annulus(0, 3)
    fat_wall = annulus(3, 6)
    collar_path = root / "collar.npz"
    np.savez_compressed(
        collar_path,
        points_world_m=collar_points,
        outer_faces=outer_faces,
        skin_sidewall_faces=skin_wall,
        fat_sidewall_faces=fat_wall,
        donor_O_parent_node_rows=np.asarray([11, 12, 13], dtype=np.uint32),
        donor_O_to_candidate_vertex=np.asarray([0, 1, 2], dtype=np.uint32),
        donor_I_parent_node_rows=np.asarray([21, 22, 23], dtype=np.uint32),
        donor_D_parent_node_rows=np.asarray([31, 32, 33], dtype=np.uint32),
        skin_sidewall_parent_tet_rows=np.arange(6, dtype=np.uint32) + 100,
        fat_sidewall_parent_tet_rows=np.arange(6, dtype=np.uint32) + 200,
        exact_candidate_face_source_rows=np.asarray([501], dtype=np.int64),
    )
    outer_path = root / "outer.npz"
    np.savez_compressed(
        outer_path,
        points_world_m=outer_points,
        source_O_faces=outer_faces,
        envelope_source_face_row=np.asarray([601], dtype=np.int64),
        envelope_source_original_id=np.asarray([7001], dtype=np.int64),
    )
    inner_path = root / "inner.npz"
    np.savez_compressed(inner_path, points_world_m=inner_points,
                        faces=outer_faces, face_source_row=np.asarray([101], dtype=np.int64),
                        face_id=np.asarray([8001], dtype=np.int64))
    deep_path = root / "deep.npz"
    np.savez_compressed(deep_path, points_world_m=deep_points,
                        faces=outer_faces, face_source_row=np.asarray([201], dtype=np.int64),
                        face_id=np.asarray([9001], dtype=np.int64))
    return outer_path, inner_path, deep_path, collar_path



def _write_tetgen_fixture(node_path: Path, ele_path: Path, points: np.ndarray,
                           node_ids: list[int], tet_node_ids: list[int], material: int) -> None:
    with node_path.open("w") as stream:
        stream.write(f"{len(node_ids)} 3 0 0\n")
        for node_id, point in zip(node_ids, points):
            stream.write(f"{node_id} {float(point[0]):.9g} {float(point[1]):.9g} {float(point[2]):.9g}\n")
    with ele_path.open("w") as stream:
        stream.write(f"1 4 1\n1 " + " ".join(map(str, tet_node_ids)) + f" {material}\n")


def _fixture_shell(points: np.ndarray, faces: np.ndarray, markers: list[int],
                   registered_i_faces: np.ndarray) -> dict[str, object]:
    count = len(faces)
    ancestry = {
        "face_ancestry_offsets": np.arange(count + 1, dtype=np.int64),
        "face_ancestry_source_rows": np.arange(700, 700 + count, dtype=np.int64),
        "face_source_role": np.asarray(["fixture_source"] * count, dtype="U24"),
        "face_source_row": np.arange(800, 800 + count, dtype=np.int64),
        "face_exact_source_row": np.arange(900, 900 + count, dtype=np.int64),
        "face_source_original_id": np.arange(1000, 1000 + count, dtype=np.int64),
        "face_source_layer": np.zeros(count, dtype=np.int64),
        "face_source_body": np.ones(count, dtype=np.int64),
        "face_source_bone": np.full(count, -1, dtype=np.int64),
        "face_ancestry_kind": np.ones(count, dtype=np.int64),
        "face_run_original_id": np.arange(1100, 1100 + count, dtype=np.int64),
        "face_source_face_id": np.arange(1200, 1200 + count, dtype=np.int64),
        "face_parent_tet_row": np.full(count, -1, dtype=np.int64),
        "face_parent_material_index": np.full(count, -1, dtype=np.int64),
        "face_parent_source_node_rows": np.full((count, 3), -1, dtype=np.int64),
    }
    return {
        "points_world_m": points.astype(np.float32), "faces": faces.astype(np.int64),
        "face_marker": np.asarray(markers, dtype=np.int32), "face_ancestry": ancestry,
        "registered_I_source_vertex_rows": np.asarray([0, 1, 2], dtype=np.int64),
        "registered_I_shell_node_rows": np.asarray([0, 1, 2], dtype=np.int64),
        "registered_I_source_faces": np.asarray([[0, 1, 2]], dtype=np.int64),
        "registered_I_shell_faces": registered_i_faces.astype(np.int64),
    }


def run_joint_tissue_fixture(root: Path) -> dict[str, object]:
    skin_points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                              [0, 0, 1], [4, 4, 4]], dtype=np.float32)
    fat_points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [0, 0, -1], [4, 4, 4]], dtype=np.float32)
    skin_node_ids = [10, 20, 30, 40, 50]
    fat_node_ids = [110, 120, 130, 140, 150]
    skin_node, skin_ele = root / "fixture-skin.node", root / "fixture-skin.ele"
    fat_node, fat_ele = root / "fixture-fat.node", root / "fixture-fat.ele"
    _write_tetgen_fixture(skin_node, skin_ele, skin_points, skin_node_ids,
                          [10, 20, 30, 40], 15)
    # This source ordering is negative in F32; the adapter must normalize it
    # before producing the child .ele file consumed by the strict native reader.
    _write_tetgen_fixture(fat_node, fat_ele, fat_points, fat_node_ids,
                          [110, 120, 130, 140], 16)
    skin_faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    fat_faces = np.asarray([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], dtype=np.int64)
    skin = _fixture_shell(skin_points[:4], skin_faces, [2, 1, 1, 1], np.asarray([[0, 2, 1]]))
    fat = _fixture_shell(fat_points[:4], fat_faces, [4, 5, 5, 5], np.asarray([[0, 1, 2]]))
    inner = {
        "points": skin_points[:3].copy(),
        "faces": np.asarray([[0, 1, 2]], dtype=np.int64),
        "source_rows": np.asarray([77], dtype=np.int64),
        "exact_source_rows": np.asarray([707], dtype=np.int64),
        "source_face_ids": np.asarray([7077], dtype=np.int64),
    }
    skin_import = module._test_tetgen_result(skin_node, skin_ele, 15, root, "fixture-skin")
    fat_import = module._test_tetgen_result(fat_node, fat_ele, 16, root, "fixture-fat")
    assert skin_import["cells_reoriented_positive"] == 0
    assert fat_import["cells_reoriented_positive"] == 1
    receipt = module.combine_tetgen_children(
        root / "fixture-skin-positive.node", root / "fixture-skin-positive.ele",
        root / "fixture-fat-positive.node", root / "fixture-fat-positive.ele",
        skin, fat, inner, root, require_registered_surface_maps=False)
    assert receipt["status"] == "COMBINED_POSITIVE_F32_CHILD_MESH_NOT_PHYSICAL_ADMISSION"
    assert receipt["registered_surface_node_map_complete"] is False
    assert receipt["nodes"] == 7 and receipt["combined_tetrahedra"] == 2
    assert receipt["registered_I_source_vertices"] == 3
    assert receipt["registered_I_source_faces"] == 1
    assert receipt["only_registered_I_nodes_merged"] is True
    assert receipt["unregistered_equal_coordinate_nodes_welded"] is False
    with gzip.open(root / "skin-fat-combined-node-map.csv.gz", "rt", newline="") as stream:
        node_rows = list(csv.DictReader(stream))
    spare = {row["source_partition"]: row for row in node_rows
             if row["source_tetgen_node_id"] in ("50", "150")}
    assert set(spare) == {"skin", "fat"}
    assert spare["skin"]["joint_node_id"] != spare["fat"]["joint_node_id"]
    assert {"registered_surface_layer", "registered_surface_vertex_row"} <= set(node_rows[0])
    assert all(row["registered_surface_layer"] in ("", "I") for row in node_rows)
    legacy_i = {}
    for row in node_rows:
        if row["registered_surface_layer"] == "I":
            legacy_i.setdefault(row["source_partition"], {})[
                int(row["registered_surface_vertex_row"])] = int(row["joint_node_id"])
    assert legacy_i["skin"] == legacy_i["fat"] == {0: 1, 1: 2, 2: 3}
    with gzip.open(root / "skin-fat-combined-element-map.csv.gz", "rt", newline="") as stream:
        cells = list(csv.DictReader(stream))
    assert [row["child_tet_row"] for row in cells] == ["0", "1"]
    assert [row["material_index"] for row in cells] == ["15", "16"]
    with gzip.open(root / "skin-fat-combined-I-interface.csv.gz", "rt", newline="") as stream:
        interface = list(csv.DictReader(stream))
    assert len(interface) == 1
    assert interface[0]["source_face_row"] == "77"
    assert interface[0]["exact_source_face_row"] == "707"
    assert interface[0]["skin_child_tet_row"] == "0"
    assert interface[0]["fat_child_tet_row"] == "1"
    receipt["source_orientation_normalization"] = {
        "skin_cells_reoriented": skin_import["cells_reoriented_positive"],
        "fat_cells_reoriented": fat_import["cells_reoriented_positive"],
    }
    combined = module.read_tetgen_mesh(root / "skin-fat-combined.node",
                                       root / "skin-fat-combined.ele", None)
    assert len(combined["tetrahedra"]) == 2
    assert set(combined["material_indices"].tolist()) == {15, 16}
    return receipt

def assert_registered_surface_maps(outer, inner, deep, skin, fat):
    def verify(surface, shell, layer, marker, role, node_offset, face_offset):
        source_points = np.asarray(surface["points"], dtype=np.float32)
        source_faces = np.asarray(surface["faces"], dtype=np.int64)
        source_vertex_rows = np.arange(len(source_points), dtype=np.int64)
        source_face_rows = np.arange(len(source_faces), dtype=np.int64)
        source_nodes = np.asarray(shell[f"registered_{layer}_source_vertex_rows"], dtype=np.int64)
        shell_nodes = np.asarray(shell[f"registered_{layer}_shell_node_rows"], dtype=np.int64)
        shell_source_faces = np.asarray(shell[f"registered_{layer}_source_faces"], dtype=np.int64)
        shell_face_rows = np.asarray(shell[f"registered_{layer}_shell_face_rows"], dtype=np.int64)
        assert np.array_equal(source_nodes, source_vertex_rows)
        assert np.array_equal(shell_nodes, source_vertex_rows + node_offset)
        assert np.array_equal(shell["points_world_m"][shell_nodes], source_points)
        assert np.array_equal(shell_source_faces, source_faces)
        assert np.array_equal(shell_face_rows, source_face_rows + face_offset)
        actual_faces = np.asarray(shell["faces"], dtype=np.int64)[shell_face_rows]
        expected_faces = source_faces + node_offset
        assert np.array_equal(np.sort(actual_faces, axis=1), np.sort(expected_faces, axis=1))
        assert np.all(np.asarray(shell["face_marker"], dtype=np.int64)[shell_face_rows] == marker)
        roles = np.asarray(shell["face_ancestry"]["face_source_role"]).astype(str)
        assert np.all(roles[shell_face_rows] == role)
        ancestry = shell["face_ancestry"]
        assert np.array_equal(np.asarray(ancestry["face_source_row"])[shell_face_rows],
                              np.asarray(surface["source_rows"]))
        assert np.array_equal(np.asarray(ancestry["face_exact_source_row"])[shell_face_rows],
                              np.asarray(surface["exact_source_rows"]))

    no, ni = len(outer["points"]), len(inner["points"])
    verify(outer, skin, "O", module.MARKER["skin_outer_O"], "outer_O", 0, 0)
    verify(inner, skin, "I", module.MARKER["skin_inner_I"], "inner_I", no, len(outer["faces"]))
    verify(inner, fat, "I", module.MARKER["fat_outer_I"], "outer_I", 0, 0)
    verify(deep, fat, "D", module.MARKER["fat_inner_D"], "inner_D", ni, len(inner["faces"]))
    assert np.array_equal(skin["registered_I_source_vertex_rows"],
                          fat["registered_I_source_vertex_rows"])
    assert np.array_equal(skin["registered_I_source_faces"],
                          fat["registered_I_source_faces"])
    assert np.array_equal(
        np.sort(skin["registered_I_shell_faces"], axis=1),
        np.sort(np.asarray(inner["faces"], dtype=np.int64) + no, axis=1))
    assert np.array_equal(
        np.sort(fat["registered_I_shell_faces"], axis=1),
        np.sort(np.asarray(inner["faces"], dtype=np.int64), axis=1))
    roles = np.asarray(skin["face_ancestry"]["face_source_role"]).astype(str)
    assert np.all(roles[len(outer["faces"]) + len(inner["faces"]):] ==
                  "donor_skin_OI_headwall")
    roles = np.asarray(fat["face_ancestry"]["face_source_role"]).astype(str)
    assert np.all(roles[len(inner["faces"]) + len(deep["faces"]):] ==
                  "donor_fat_ID_headwall")


def run_fixture() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="native-domain-217-") as tmp:
        root = Path(tmp)
        outer_path, inner_path, deep_path, collar_path = save_fixture_npzs(root)
        collar = np.load(collar_path, allow_pickle=False)
        outer = module.load_surface(outer_path, "O", collar)
        inner = module.load_surface(inner_path, "I")
        deep = module.load_surface(deep_path, "D")
        assert outer["exact_source_rows"].tolist() == [501]
        assert inner["ancestry"] == [[101]] and deep["ancestry"] == [[201]]
        skin_walls = np.arange(6, dtype=np.int64) + 100
        fat_walls = np.arange(6, dtype=np.int64) + 200
        wall_rows = {
            "skin_parent_tets": skin_walls,
            "skin_parent_nodes": np.tile(np.asarray([[11, 12, 22]], dtype=np.int64), (6, 1)),
            "skin_materials": np.full(6, 15, dtype=np.int64),
            "fat_parent_tets": fat_walls,
            "fat_parent_nodes": np.tile(np.asarray([[21, 22, 32]], dtype=np.int64), (6, 1)),
            "fat_materials": np.full(6, 16, dtype=np.int64),
        }
        skin, fat, assembly = module.assemble_shells(outer, inner, deep, collar, wall_rows)
        assert_registered_surface_maps(outer, inner, deep, skin, fat)
        for label, shell in (("skin", skin), ("fat", fat)):
            edge = module.edge_report(shell["faces"])
            assert edge["boundary_edges"] == 0 and edge["nonmanifold_edges"] == 0
            assert edge["same_direction_twoface_edges"] == 0
            assert shell["shell_signed_volume_m3"] > 0
            assert len(shell["face_marker"]) == len(shell["faces"])
            assert len(shell["face_ancestry"]["face_source_role"]) == len(shell["faces"])
            module.write_shell_npz(root / f"{label}.npz", shell, label)
            written_shell = np.load(root / f"{label}.npz", allow_pickle=False)
            assert "registered_I_source_vertex_rows" in written_shell.files
            assert "registered_I_shell_node_rows" in written_shell.files
            assert "registered_I_source_faces" in written_shell.files
            assert "registered_I_shell_face_rows" in written_shell.files
            if label == "skin":
                assert "registered_O_source_vertex_rows" in written_shell.files
            if label == "fat":
                assert "registered_D_source_vertex_rows" in written_shell.files
            module.write_face_ancestry_csv(root / f"{label}-faces.csv.gz", shell)
        assert set(skin["face_marker"].tolist()) == {1, 2, 3}
        assert set(fat["face_marker"].tolist()) == {4, 5, 6}
        skin_poly = root / "skin.poly"
        fat_poly = root / "fat.poly"
        module.write_poly(skin_poly, skin["points_world_m"], skin["faces"],
                          skin["face_marker"], [np.asarray([0, 0, .5])],
                          [(np.asarray([0, 0, 1.5]), 15)])
        module.write_poly(fat_poly, fat["points_world_m"], fat["faces"],
                          fat["face_marker"], [np.asarray([0, 0, .5])],
                          [(np.asarray([0, 0, .5]), 16)])
        tet_points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        _, positive, volumes = module.positive_f32_tetrahedra(
            tet_points, np.asarray([[0, 2, 1, 3]], dtype=np.int64), 15)
        assert sorted(positive[0].tolist()) == [0, 1, 2, 3] and np.all(volumes > 0)
        collapsed = tet_points.copy()
        collapsed[3] = collapsed[0]
        try:
            module.positive_f32_tetrahedra(collapsed, np.asarray([[0, 1, 2, 3]]), 15)
        except ValueError:
            pass
        else:
            raise AssertionError("Float32-collapsed tetrahedron was accepted")
        joint = run_joint_tissue_fixture(root)
        return {
            "status": "FIXTURE_PASS_NOT_PHYSICAL_ADMISSION",
            "assembled_O_I_D_source_point_face_and_role_maps_exact": True,
            "skin_faces": len(skin["faces"]),
            "fat_faces": len(fat["faces"]),
            "skin_signed_volume_m3": skin["shell_signed_volume_m3"],
            "fat_signed_volume_m3": fat["shell_signed_volume_m3"],
            "poly_sections_valid": True,
            "positive_f32_tet_and_collapse_rejection": True,
            "joint_mesh": joint,
            "physical_admission": False,
        }


if __name__ == "__main__":
    import json
    print(json.dumps(run_fixture(), sort_keys=True))
