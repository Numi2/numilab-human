"""Check the registered gluteal solids before using them for native load transfer."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

from audit import BONE, accepted_pose, closed_surface, digest, fem_boundaries, records, rows
from audit_bed_layer_surfaces import install_for_float32, owner, role_surfaces
from check_candidate import targets
from extension import determinant, quality
from prepare_batch import parse_tetgen, write_poly
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from register_tissue_bellies import source, blend


def crossings(mesh, faces, other, identify):
    use = targets(other, (mesh.min(axis=0), mesh.max(axis=0)))
    triangles = other[use]
    a, _ = owner.float32_surface_records(mesh, faces)
    b, _ = owner.float32_surface_records(
        triangles.reshape(-1, 3), np.arange(triangles.size // 3).reshape(-1, 3))
    pairs = owner._audit_pair_prepared_first(owner._prepare_surface_aabb(a), b)["triangle_pairs"]
    return {
        "candidate_triangles": len(use),
        "pairs": len(pairs),
        "witnesses": [{"gluteal_source_face": int(i), "target": identify(use[j])}
                      for i, j in pairs],
    }


def surface_volume(points, faces):
    face = points[faces] - points.mean(axis=0)
    a, b, c = np.moveaxis(face, 1, 0)
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.)


def shape(bank, pose):
    world, _ = blend(bank, pose)
    points, parent, index = np.unique(world.astype(np.float32), axis=0, return_index=True,
                                       return_inverse=True)
    faces = index[bank["source_faces"]]
    assert closed_surface(points, faces)
    edge = np.sort(faces[:, (0, 1, 1, 2, 2, 0)].reshape(-1, 2), axis=1)
    _, owners = np.unique(edge, axis=0, return_inverse=True)
    pair = np.argsort(owners).reshape(-1, 2) // 3
    graph = coo_matrix((np.ones(len(pair)), (pair[:, 0], pair[:, 1])),
                       shape=(len(faces), len(faces))).tocsr()
    components, labels = connected_components(graph, directed=False)
    points = points.astype(float)
    assert surface_volume(points, faces) > 0.
    return points, faces, bank["row_ids"][parent], [faces[labels == i] for i in range(components)]


def volume_model(tetgen, prefix, points, faces, parent):
    poly = prefix.with_suffix(".poly")
    write_poly(poly, points, faces)
    check = subprocess.run([tetgen, "-d", poly.name], cwd=poly.parent, capture_output=True,
                           text=True, timeout=30)
    poly.with_suffix(".intersection.log").write_text(check.stdout + check.stderr)
    if check.returncode or "The input surface mesh is correct." not in check.stdout:
        return {"surface_check_returncode": check.returncode, "surface_check_passed": False}
    build = subprocess.run([tetgen, "-pYq1.5n", poly.name], cwd=poly.parent, capture_output=True,
                           text=True, timeout=75)
    poly.with_suffix(".meshing.log").write_text(build.stdout + build.stderr)
    result = {"surface_check_passed": True, "volume_build_returncode": build.returncode}
    if build.returncode:
        return result
    node = parse_tetgen(prefix.parent / (prefix.name + ".1.node"), "node")
    tets = parse_tetgen(prefix.parent / (prefix.name + ".1.ele"), "ele")
    boundary = parse_tetgen(prefix.parent / (prefix.name + ".1.face"), "face")
    nodes = np.array([p for _, p, _ in node], dtype=np.float32)
    remap = {n: i for i, (n, _, _) in enumerate(node)}
    cells = np.array([[remap[v] for v in verts] for _, verts in tets])
    marker = np.array([m for _, _, m in boundary])
    assert np.array_equal(np.sort(marker), np.arange(1, len(faces) + 1))
    for _, vertices, ancestry in boundary:
        actual = nodes[[remap[v] for v in vertices]].tolist()
        assert sorted(actual) == sorted(points[faces[ancestry - 1]].tolist())
    nodes = nodes.astype(float)
    reference_volume = surface_volume(points, faces)
    det = determinant(nodes, cells)
    metric = quality(nodes, cells, det)
    result.update(nodes=len(nodes), tets=len(cells), signed_volume_m3=float(det.sum() / 6.),
                  surface_volume_m3=reference_volume,
                  volume_relative_error=float(abs(det.sum() / 6. - reference_volume) / reference_volume),
                  nonpositive_cells=int((det <= 0).sum()),
                  minimum_signed_determinant_m3=float(det.min()),
                  determinant_over_edge_cubed_quantiles=np.quantile(metric, (0, .01, .1, .5)).tolist(),
                  boundary_parent_markers_verified=True)
    np.savez_compressed(prefix.with_suffix(".npz"), nodes=nodes, cells=cells, source_nodes=points,
                        source_faces=faces, source_vertex_ids=parent)
    return result


def build(run, candidate, bone_path, tetgen, out):
    install_for_float32()
    launch = json.loads((run / "launch.json").read_text())
    argv = launch["argv"]
    payload = Path(argv[argv.index("--soft-tissue-payload") + 1])
    tissue_hash = digest(payload)
    manifest = json.loads(payload.with_suffix(".manifest.json").read_text())
    assert manifest["payload"]["sha256"] == tissue_hash
    labels = {r["stable_id"]: r.get("label") or r["source_name"] for r in manifest["source"]["surfaces"]}

    nodes, anchors, cells, evidence = records(run / "stdout.log")
    reference = json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2])
    poses, pose_check = accepted_pose(run / "native/resting-body-motion-partition.csv", reference)
    tissues, metadata = fem_boundaries(nodes, cells, anchors)
    shell_faces = role_surfaces(rows(run / "shell-mass.csv.tets.csv"))
    with np.load(candidate) as fitted:
        shell = fitted["nodes_current"]
    baseline = np.array([[v[f"current_{axis}_m"] for axis in "xyz"]
                         for v in rows(run / "shell-mass.csv.nodes.csv")], dtype=np.float32)
    fitted_surfaces = np.array([face for group in shell_faces.values() for face in group])
    fitted_roles = [role for role, group in shell_faces.items() for _ in group]
    bones = np.memmap(bone_path, dtype=BONE, mode="r")
    banks = source(payload)
    report = {
        "tissue_sha256": tissue_hash, "bone_sha256": digest(bone_path),
        "shell_sha256": digest(candidate), "tetgen_sha256": digest(tetgen), "pose": pose_check,
        "scope": "Registered Float32 source surface at first accepted rigid pose. Tests boundary crossings and standalone volume geometry, excluding containment and native dynamics.",
        "banks": [],
    }
    for stable_id in (29, 30):
        bank = banks[stable_id]
        points, faces, parent, parts = shape(bank, poses)
        result = {"stable_id": stable_id, "label": labels[stable_id],
                  "bound_bodies": [binding[0] for binding in bank["bindings"]],
                  "source_vertices": len(points), "source_faces": len(faces),
                  "components": [{"faces": len(part), "signed_volume_m3": surface_volume(points, part)}
                                 for part in parts]}
        result["bone"] = crossings(
            points, faces, bones["vertices"],
            lambda i: {key: int(bones[key][i]) for key in ("body", "bone", "triangle")})
        result["bone"]["pairs_by_bank"] = dict(Counter(
            f'{v["target"]["body"]}/{v["target"]["bone"]}' for v in result["bone"]["witnesses"]))
        for tag, boundary in (("skin_fat_baseline", baseline), ("skin_fat", shell)):
            result[tag] = crossings(
                points, faces, boundary[fitted_surfaces],
                lambda i: {"role": fitted_roles[i], "nodes": fitted_surfaces[i].tolist()})
            result[tag]["pairs_by_role"] = dict(Counter(
                v["target"]["role"] for v in result[tag]["witnesses"]))
        result["other_fem"] = crossings(points, faces, tissues, metadata.__getitem__)
        result["volumetric"] = volume_model(tetgen, out / f"gluteal-{stable_id}", points, faces, parent)
        report["banks"].append(result)
        print(json.dumps({k: {a: b for a, b in v.items() if a != "witnesses"}
                          if isinstance(v, dict) else v for k, v in result.items()}), flush=True)
    return report


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for arg in ("run", "candidate", "bone", "tetgen"):
        p.add_argument(arg, type=Path)
    p.add_argument("--out", type=Path, default=Path.cwd())
    a = p.parse_args()
    report = build(a.run, a.candidate, a.bone, a.tetgen, a.out)
    with (a.out / "gluteal-registration.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
