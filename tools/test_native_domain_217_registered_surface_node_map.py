from __future__ import annotations

import csv
import gzip
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("build_native_domain_217.py")
spec = importlib.util.spec_from_file_location("native_domain_217_node_map_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def write_mesh(node_path, ele_path, points, node_ids, element_ids, tets, material):
    with node_path.open("w") as stream:
        stream.write(f"{len(node_ids)} 3 0 0\n")
        for node, point in zip(node_ids, points):
            stream.write(f"{node} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
    with ele_path.open("w") as stream:
        stream.write(f"{len(element_ids)} 4 1\n")
        for element, tet in zip(element_ids, tets):
            corners = [node_ids[int(v)] for v in tet]
            stream.write(f"{element} " + " ".join(map(str, corners)) + f" {material}\n")


def ancestry(count, roles):
    return {
        "face_ancestry_offsets": np.arange(count + 1, dtype=np.int64),
        "face_ancestry_source_rows": np.arange(700, 700 + count, dtype=np.int64),
        "face_source_role": np.asarray(roles, dtype="U24"),
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


def prism_shell(points, node_path, ele_path, material, caps, interface_faces):
    mesh = module.read_tetgen_mesh(node_path, ele_path, material)
    boundary = module._tetrahedron_boundary_faces(mesh)
    faces, markers, roles = [], [], []
    face_rows = {name: [] for name in caps}
    for entries in boundary.values():
        if len(entries) == 2:
            continue
        assert len(entries) == 1
        tri = np.asarray(entries[0]["oriented_nodes"], dtype=np.int64)
        matches = []
        for name, cap in caps.items():
            cap_nodes = np.asarray(cap["shell_nodes"], dtype=np.int64)
            cap_faces = np.asarray(cap["source_faces"], dtype=np.int64)
            mapped = cap_nodes[cap_faces[0]]
            if tuple(sorted(map(int, tri))) == tuple(sorted(map(int, mapped))):
                matches.append(name)
        assert len(matches) <= 1
        if matches:
            name = matches[0]
            marker, role = caps[name]["marker"], caps[name]["role"]
            face_rows[name].append(len(faces))
        else:
            marker, role = ((module.MARKER["skin_wall_OI"], "donor_skin_OI_headwall")
                            if material == 15 else
                            (module.MARKER["fat_wall_ID"], "donor_fat_ID_headwall"))
        faces.append(tri.tolist())
        markers.append(marker)
        roles.append(role)

    count = len(faces)
    shell = {
        "points_world_m": np.asarray(points, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int64),
        "face_marker": np.asarray(markers, dtype=np.int32),
        "face_ancestry": ancestry(count, roles),
        "registered_I_source_vertex_rows": np.arange(3, dtype=np.int64),
        "registered_I_shell_node_rows": np.asarray(caps["I"]["shell_nodes"], dtype=np.int64),
        "registered_I_source_faces": np.asarray(caps["I"]["source_faces"], dtype=np.int64),
        "registered_I_shell_faces": np.asarray(caps["I"]["shell_nodes"], dtype=np.int64)[
            np.asarray(caps["I"]["source_faces"], dtype=np.int64)],
        "registered_I_shell_face_rows": np.asarray(face_rows["I"], dtype=np.int64),
        "registered_I_source_face_rows": np.asarray([77], dtype=np.int64),
        "registered_I_source_exact_face_rows": np.asarray([707], dtype=np.int64),
        "registered_I_source_face_ids": np.asarray([7077], dtype=np.int64),
        "registered_I_source_ancestry_offsets": np.asarray([0, 1], dtype=np.int64),
        "registered_I_source_ancestry_rows": np.asarray([77], dtype=np.int64),
    }
    for name in ("O", "D"):
        if name in caps:
            shell[f"registered_{name}_source_vertex_rows"] = np.arange(3, dtype=np.int64)
            shell[f"registered_{name}_shell_node_rows"] = np.asarray(
                caps[name]["shell_nodes"], dtype=np.int64)
            shell[f"registered_{name}_source_faces"] = np.asarray(
                caps[name]["source_faces"], dtype=np.int64)
            shell[f"registered_{name}_shell_face_rows"] = np.asarray(
                face_rows[name], dtype=np.int64)
    return shell


def main():
    with tempfile.TemporaryDirectory(prefix="native-domain-217-node-map-") as tmp:
        root = Path(tmp)
        base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        inner_points = base.copy()
        outer_points = base + np.asarray([0, 0, 1], dtype=np.float32)
        deep_points = base + np.asarray([0, 0, -1], dtype=np.float32)
        skin_points = np.vstack((outer_points, inner_points)).astype(np.float32)
        fat_points = np.vstack((inner_points, deep_points)).astype(np.float32)
        tets = np.asarray([[3, 4, 5, 2], [3, 4, 2, 1], [3, 0, 1, 2]], dtype=np.int64)
        node_ids = [47, 11, 83, 5, 71, 29]
        sf = root / "skin.node", root / "skin.ele"
        ff = root / "fat.node", root / "fat.ele"
        write_mesh(*sf, skin_points, node_ids, [901, 703, 845], tets, 15)
        write_mesh(*ff, fat_points, node_ids, [509, 407, 601], tets, 16)
        skin_import = module._test_tetgen_result(*sf, 15, root, "prism-skin")
        fat_import = module._test_tetgen_result(*ff, 16, root, "prism-fat")
        assert skin_import["positive_f32_cells"] == fat_import["positive_f32_cells"] == 3

        tri = np.asarray([[0, 1, 2]], dtype=np.int64)
        skin = prism_shell(
            skin_points, root / "prism-skin-positive.node", root / "prism-skin-positive.ele",
            15, {
                "O": {"marker": module.MARKER["skin_outer_O"], "role": "outer_O",
                      "shell_nodes": np.arange(3), "source_faces": tri},
                "I": {"marker": module.MARKER["skin_inner_I"], "role": "inner_I",
                      "shell_nodes": np.arange(3, 6), "source_faces": tri},
            }, tri)
        fat = prism_shell(
            fat_points, root / "prism-fat-positive.node", root / "prism-fat-positive.ele",
            16, {
                "I": {"marker": module.MARKER["fat_outer_I"], "role": "outer_I",
                      "shell_nodes": np.arange(3), "source_faces": tri},
                "D": {"marker": module.MARKER["fat_inner_D"], "role": "inner_D",
                      "shell_nodes": np.arange(3, 6), "source_faces": tri},
            }, tri)
        inner = {"points": inner_points, "faces": tri, "source_rows": np.asarray([77]),
                 "exact_source_rows": np.asarray([707]), "source_face_ids": np.asarray([7077])}
        out = root / "combined"
        out.mkdir()
        receipt = module.combine_tetgen_children(
            root / "prism-skin-positive.node", root / "prism-skin-positive.ele",
            root / "prism-fat-positive.node", root / "prism-fat-positive.ele",
            skin, fat, inner, out)
        assert receipt["registered_surface_node_map_complete"] is True
        assert receipt["registered_surface_node_rows"] == {"O": 3, "I": 3, "D": 3}
        assert receipt["nodes"] == 9 and receipt["combined_tetrahedra"] == 6
        with gzip.open(out / "skin-fat-combined-node-map.csv.gz", "rt", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_layer = {}
        for row in rows:
            if row["registered_surface_layer"]:
                by_layer.setdefault(row["registered_surface_layer"], []).append(row)
        assert {key: len(value) for key, value in by_layer.items()} == {"O": 3, "I": 6, "D": 3}
        assert {int(row["registered_surface_vertex_row"]) for row in by_layer["O"]} == {0, 1, 2}
        assert {int(row["registered_surface_vertex_row"]) for row in by_layer["D"]} == {0, 1, 2}
        i_rows = {}
        for row in by_layer["I"]:
            i_rows.setdefault(row["source_partition"], {})[
                int(row["registered_surface_vertex_row"])] = int(row["joint_node_id"])
        assert i_rows["skin"] == i_rows["fat"] == {0: 4, 1: 5, 2: 6}
        assert [int(row["source_tetgen_node_id"]) for row in rows
                if row["source_partition"] == "skin"] == node_ids
        assert [int(row["source_tetgen_node_id"]) for row in rows
                if row["source_partition"] == "fat"] == node_ids

        def rejects(change, name, reason):
            bad_skin = {**skin, "face_ancestry": {**skin["face_ancestry"],
                "face_source_role": skin["face_ancestry"]["face_source_role"].copy()}}
            bad_fat = {**fat, "face_ancestry": {**fat["face_ancestry"],
                "face_source_role": fat["face_ancestry"]["face_source_role"].copy()}}
            change(bad_skin, bad_fat)
            try:
                module.combine_tetgen_children(
                    root / "prism-skin-positive.node", root / "prism-skin-positive.ele",
                    root / "prism-fat-positive.node", root / "prism-fat-positive.ele",
                    bad_skin, bad_fat, inner, root / f"negative-{name}")
            except ValueError as error:
                assert reason in str(error), (name, str(error))
            else:
                raise AssertionError(f"accepted invalid registered layer map: {name}")

        rejects(lambda s, d: s.pop("registered_O_shell_node_rows"),
                "missing-O", "missing its exact registered-O")
        rejects(lambda s, d: s.__setitem__("registered_O_source_vertex_rows",
                    np.asarray([0, 1, 1], dtype=np.int64)),
                "duplicate-O", "missing, duplicate, or invalid registered-O node row")
        rejects(lambda s, d: s.__setitem__("registered_O_shell_node_rows",
                    np.asarray([0, 0, 2], dtype=np.int64)),
                "collapsed-O", "missing, duplicate, or invalid registered-O node row")
        rejects(lambda s, d: d.pop("registered_D_source_faces"),
                "missing-D", "missing its exact registered-D")
        rejects(lambda s, d: d.__setitem__("registered_D_source_vertex_rows",
                    np.asarray([0, 1, 1], dtype=np.int64)),
                "duplicate-D", "missing, duplicate, or invalid registered-D node row")
        rejects(lambda s, d: d.__setitem__("registered_D_shell_node_rows",
                    np.asarray([3, 3, 5], dtype=np.int64)),
                "collapsed-D", "missing, duplicate, or invalid registered-D node row")
        rejects(lambda s, d: s["face_ancestry"]["face_source_role"].__setitem__(
                    int(s["registered_O_shell_face_rows"][0]), "wrong-role"),
                "missing-O-role", "face markers/roles are missing")
        rejects(lambda s, d: d["face_ancestry"]["face_source_role"].__setitem__(
                    int(d["registered_D_shell_face_rows"][0]), "wrong-role"),
                "missing-D-role", "face markers/roles are missing")

        print("PASS synthetic prism O/I/D node-row maps; 9 combined nodes, 6 positive F32 cells; "
              "reordered TetGen IDs; exact shared I; 8 fail-closed O/D mapping cases; no physical admission")


if __name__ == "__main__":
    main()
