"""Construct and independently audit a common medial partition of the native gluteal inputs."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np
from igl.copyleft import cgal

from audit import BONE, accepted_pose, digest, fem_boundaries, muscle_geometry, records, rows
from audit_bed_layer_surfaces import install_for_float32, owner, role_surfaces
from check_candidate import targets
from fit_muscle_interface import exact_pairs, MARGIN_M, prepare_target
from verify_containment import classify, surface


BOX_TRIANGLES = np.array([
    (0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6),
    (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3),
    (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5),
], dtype=np.int64)[:, [0, 2, 1]]


def surface_volume(points, faces):
    # Oriented flux through the closed boundary in metres cubed.
    return float(np.linalg.det(points[faces] - points.mean(axis=0)).sum() / 6.)


def boundary_frame(log):
    with log.open() as f:
        state = json.loads(next(row.partition("=")[2] for row in f
                                if row.startswith("resting_initial_anatomical_registration=")))
    by_id = {b["body_index"]: np.array(b["position_m"]) for b in state["body_poses"]}
    origin = (by_id[131] + by_id[145]) / 2
    axis = by_id[145] - by_id[131]
    axis /= np.linalg.norm(axis)
    fore = np.array([0., 1., 0.]) - axis * axis[1]
    fore /= np.linalg.norm(fore)
    basis = np.array([axis, fore, np.cross(axis, fore)])
    assert np.linalg.det(basis) > 0.
    return origin, basis


def halfspace_proxy(local, origin, basis, sign):
    low, high = local.min(axis=0) - .025, local.max(axis=0) + .025
    if sign < 0:
        high[0] = -MARGIN_M / 2
        cap = [2, 3]
    else:
        low[0] = MARGIN_M / 2
        cap = [0, 1]
    coords = np.array([(x, y, z) for x in (low[0], high[0])
                      for y in (low[1], high[1]) for z in (low[2], high[2])])
    result = coords @ basis + origin
    assert surface_volume(result, BOX_TRIANGLES) > 0
    return result, cap


def weld_emitted(unrounded, faces, parents):
    """Contract coincident coordinates only along connected edges of the cut."""
    points = unrounded.astype(np.float32)
    vertices, inverse, counts = np.unique(points, axis=0, return_inverse=True,
                                           return_counts=True)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    coincident = edges[inverse[edges[:, 0]] == inverse[edges[:, 1]]]
    for group in np.flatnonzero(counts > 1):
        members = set(np.flatnonzero(inverse == group))
        links = coincident[inverse[coincident[:, 0]] == group]
        seen = {min(members)}
        while not members <= seen:
            expanded = seen | {v for a, b in links if a in seen or b in seen for v in (a, b)}
            assert expanded != seen, "distinct components share a rounded coordinate"
            seen = expanded
    mapped = inverse[faces]
    removed = ((mapped[:, 0] == mapped[:, 1]) | (mapped[:, 1] == mapped[:, 2])
               | (mapped[:, 0] == mapped[:, 2]))
    # A clipped descendant with zero emitted area carries no native surface.
    # Record every original parent and pre-emission area affected by this
    # contraction. Exact closure and all neighbor tests still run below.
    zero = unrounded[faces[removed]]
    area = .5 * np.linalg.norm(np.cross(zero[:, 1] - zero[:, 0], zero[:, 2] - zero[:, 0]), axis=1)
    collapsed = dict(faces=int(removed.sum()), total_unrounded_area_m2=float(area.sum()),
                     source_face_children=sorted(map(int, parents[removed & (parents >= 0)])))
    return vertices, mapped[~removed], parents[~removed], collapsed


def clip(path, origin, basis, sign, envelope):
    with np.load(path) as data:
        base = data["source_nodes"].astype(float)
        old_faces = data["source_faces"]
        source_ids = data["source_vertex_ids"]
    proxy, cap = halfspace_proxy((base - origin) @ basis.T, origin, basis, sign)
    verts, faces, parents = cgal.mesh_boolean(base, old_faces, proxy, BOX_TRIANGLES, "intersection")
    is_cap = parents >= len(old_faces)
    assert set(parents[is_cap] - len(old_faces)) <= set(cap)
    parent_face = np.where(is_cap, -1, parents)
    if envelope is not None:
        env_points, env_faces = envelope
        verts, faces, rows = cgal.mesh_boolean(verts, faces, env_points, env_faces, "minus")
        # Negative ids distinguish the generated medial and obstacle surfaces.
        # Facet ids retain the exact input face source when preserved by both cuts.
        parent_face = np.r_[parent_face, np.full(len(env_faces), -2)][rows]
    verts, faces, parent_face, collapsed = weld_emitted(
        verts, faces, parent_face)
    signed_distances = ((verts.astype(float) - origin) @ basis[0]) * sign
    assert signed_distances.min() >= MARGIN_M / 2 - 1e-7
    base_lookup = {tuple(p): i for i, p in enumerate(base)}
    retained = np.array([base_lookup.get(tuple(v), -1) for v in verts])
    ancestry = np.full(len(verts), -1, dtype=np.int64)
    valid = retained >= 0
    ancestry[valid] = source_ids[retained[valid]]
    far = ((base - origin) @ basis[0]) * sign >= MARGIN_M / 2 + 1e-7
    if envelope is not None:
        # The envelope is the only additional region allowed to alter a node.
        env_points = envelope[0]
        far &= np.any((base < env_points.min(axis=0)) | (base > env_points.max(axis=0)), axis=1)
    assert set(np.flatnonzero(far)) <= set(retained)
    emitted = verts[faces].astype(float)
    area = np.linalg.norm(np.cross(emitted[:, 1] - emitted[:, 0],
                                  emitted[:, 2] - emitted[:, 0]), axis=1)
    print("Float32 cut emission", path.name, "welded faces", collapsed,
          "zero-area faces", int((area == 0).sum()), flush=True)
    assert np.isfinite(area).all() and area.min() > 0
    return base, old_faces, verts, faces, ancestry, parent_face, collapsed


def analyze(path, origin, basis, sign, obstacles, out, envelope, audited):
    if audited is None:
        base, old, posed, cut, indices, parent, collapsed = clip(path, origin, basis, sign, envelope)
    else:
        with np.load(path) as control, np.load(audited / path.name) as data:
            base, old = control["source_nodes"], control["source_faces"]
            posed, cut, indices, parent = (data[k] for k in ("source_nodes", "source_faces",
                                                           "source_vertex_ids", "source_face_parents"))
        collapsed = None
        side = ((posed.astype(float)-origin) @ basis[0]) * sign
        assert side.min() >= MARGIN_M / 2 - 1e-7

    surface_model = surface(posed, cut)
    records, _ = owner.float32_surface_records(posed, cut)
    self_pairs = owner._audit_self_pairs_touching_faces(records, set(range(len(cut))))["triangle_pairs"]
    before = exact_pairs(base.astype(np.float32), old, obstacles)
    after = exact_pairs(posed, cut, obstacles)
    canonical = {(owner_name, int(parent[face]), target) for owner_name, face, target in after}
    new_pairs = canonical - before
    refvol = surface_volume(base, old)
    volume = surface_volume(posed.astype(float), cut)
    r = dict(rounded_faces_collapsed=collapsed, vertices=len(posed), faces=len(cut), caps=int((parent < 0).sum()),
             non_source_vertices=int((indices < 0).sum()),
             baseline_volume_m3=refvol, prepared_volume_m3=volume,
             relative_volume_change=volume / refvol - 1.,
             self_crossing_pairs=self_pairs,
             original_pairs=dict(Counter(tag for tag, _, _ in before)),
             remaining_pairs_by_source=dict(Counter(tag for tag, _, _ in canonical)),
             new_pairs_by_source=dict(Counter(tag for tag, _, _ in new_pairs)),
             new_pairs=sorted(new_pairs))
    np.savez_compressed(out / path.name, source_nodes=posed, source_faces=cut,
                        source_vertex_ids=indices, source_face_parents=parent)
    return r, posed, surface_model, records


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("reference", "bone", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--envelope", type=Path)
    parser.add_argument("--obstacles", type=Path)
    parser.add_argument("--tissue-reader", type=Path)
    parser.add_argument("--audit-mesh", type=Path)
    a = parser.parse_args()
    assert a.envelope is None or a.obstacles is None
    envelope = None
    if a.envelope is not None:
        with np.load(a.envelope) as data:
            assert str(data["source_sha256"]) == digest(a.bone)
            envelope = (data["vertices"], data["faces"])
    install_for_float32()
    a.output.mkdir()
    origin, basis = boundary_frame(a.reference / "paired-native-128/stdout.log")
    glutes = a.reference / "bilateral-05"
    shells = a.reference / "paired-06/posterior-interface-fit.npz"
    bones = np.memmap(a.bone, dtype=BONE, mode="r")
    cell_file = a.reference / "paired-native-128/shell-mass.csv.tets.csv"
    roles = role_surfaces(rows(cell_file))
    with np.load(shells) as shell:
        shell_triangles = shell["nodes_current"][[face for group in roles.values() for face in group]]
    domain = dict(shell=shell_triangles, bones=bones["vertices"])
    if a.tissue_reader is not None:
        run = a.reference / "paired-native-128"
        nodes, anchors, cells, evidence = records(run / "stdout.log")
        registration = json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2])
        pose, _ = accepted_pose(run / "native/resting-body-motion-partition.csv", registration)
        scene, _, _, _, starts = muscle_geometry(json.loads((run / "launch.json").read_text()),
                                                  pose, a.tissue_reader)
        offsets = sorted(starts.values()) + [len(scene)]
        active = np.ones(len(scene), dtype=bool)
        for i in (29, 30):
            offset = starts[i]
            active[offset:offsets[offsets.index(offset)+1]] = False
        domain["muscles"] = scene[active]
        domain["other_fem"], _ = fem_boundaries(nodes, cells, anchors)
    output = {}
    obstacle_identities = {}
    for muscle, sign in ((29, -1), (30, 1)):
        path = glutes / f"gluteal-{muscle}.npz"
        with np.load(path) as mesh:
            pts = mesh["source_nodes"]
        prepared = {k: (ix, prepare_target(tri, ix)) for k, tri in domain.items()
                    if len(ix := targets(tri, (pts.min(axis=0), pts.max(axis=0))))}
        if a.obstacles is not None:
            cut_path = a.obstacles / f"obstacles-{muscle}.npz"
            obstacle_identities[str(muscle)] = digest(cut_path)
            with np.load(cut_path) as data:
                assert str(data["source_sha256"]) == digest(a.bone)
                assert str(data["shell_sha256"]) == digest(shells)
                envelope = (data["vertices"], data["faces"])
        output[muscle] = analyze(path, origin, basis, sign, prepared, a.output, envelope, a.audit_mesh)
    right, left = output[29], output[30]
    pairs = owner._audit_pair_prepared_first(owner._prepare_surface_aabb(right[3]), left[3])["triangle_pairs"]
    overlap = [classify(side[1], other[2]) for side, other in ((right, left), (left, right))]
    result = dict(origin_m=origin.tolist(), lateral_axis=basis[0].tolist(), nominal_separation_m=MARGIN_M,
                  exact_bilateral_pairs=pairs, contained_related_vertices=overlap,
                  details={str(i): output[i][0] for i in output},
                  bone_sha256=digest(a.bone), shell_sha256=digest(shells))
    if obstacle_identities:
        result["posterior_obstacle_hashes"] = obstacle_identities
    if a.envelope is not None:
        result["sacral_envelope_sha256"] = digest(a.envelope)
    with (a.output / "assessment.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    for i, data in result["details"].items():
        print(i, {k: v for k, v in data.items() if k != "new_pairs"})
    print("bilateral crossings", len(pairs), "contained or boundary", list(map(len, overlap)))


if __name__ == "__main__":
    main()
