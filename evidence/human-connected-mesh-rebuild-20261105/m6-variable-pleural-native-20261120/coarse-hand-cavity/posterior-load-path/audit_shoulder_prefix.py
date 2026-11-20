"""Check the retained superior shoulder attachments in accepted native poses."""
from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import audit_actual_pose_registered_ligament_catalog as audit


def load_local_meshes(payload, catalog, vertex_offset, index_offset):
    meshes = {}
    for stable_id, entry in catalog.items():
        source = np.frombuffer(payload, dtype="<f4", count=6 * entry["vertexCount"],
                               offset=vertex_offset + 24 * entry["firstVertex"]).reshape(-1, 6)
        points = audit.qrot(entry["boneQ"], source[:, :3].astype(float) * entry["scale"])
        points += np.asarray(entry["translation"])
        faces = np.frombuffer(payload, dtype="<u4", count=entry["indexCount"],
                              offset=index_offset + 4 * entry["firstIndex"]).astype(np.int32)
        faces = faces.reshape(-1, 3) - entry["firstVertex"]
        normals = audit.qrot(entry["boneQ"], source[:, 3:].astype(float))
        normal_length = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(normal_length[:, None], 1e-30)
        meshes[stable_id] = (points, faces, normals, points.min(axis=0), points.max(axis=0))
    return meshes


def row_pose(row):
    xyz = [float(row[f"root_{axis}_m"]) + float(row[f"com_relative_root_{axis}_m"]) for axis in "xyz"]
    return np.asarray(xyz + [float(row[f"orientation_{axis}"]) for axis in "xyzw"])


def local_point(pose, point):
    return audit.qrot(pose[3:] * [-1, -1, -1, 1], point - pose[:3])


def path_hits(path, pose, catalog, meshes):
    hits = []
    for stable_id, (points, faces, _, lo, hi) in meshes.items():
        body = pose[catalog[stable_id]["body"]]
        a_local, b_local = (local_point(body, p) for p in path)
        if np.all(hi >= np.minimum(a_local, b_local) - 1e-9) and np.all(
            lo <= np.maximum(a_local, b_local) + 1e-9
        ):
            hits.extend([stable_id, int(face), float(t)]
                        for t, face in audit.lp.vector_hits(a_local, b_local, (points, faces)))
    return hits


def check_path(attachment, pose, catalog, meshes):
    path, exact_path, path_normals, surface_errors = [], [], [], []
    stable_ids = attachment["stable_ids"]
    triangle_rows = attachment["triangle_rows"]
    for stable_id, face, weights, local in zip(
        stable_ids, triangle_rows, attachment["material_barycentric"],
        attachment["endpoint_body_local_m"]
    ):
        points, faces, normals, *_ = meshes[stable_id]
        coordinates = np.asarray(weights) @ points[faces[face]]
        normal = np.asarray(weights) @ normals[faces[face]]
        assert np.linalg.norm(normal) > 0
        local = np.asarray(local, dtype=np.float32).astype(float)
        surface_errors.append(float(np.linalg.norm(coordinates - local)))
        body = pose[catalog[stable_id]["body"]]
        path.append(body[:3] + audit.qrot(body[3:], local))
        exact_path.append(body[:3] + audit.qrot(body[3:], coordinates))
        path_normals.append(audit.qrot(body[3:], normal / np.linalg.norm(normal)))
    a, b = path
    delta = b - a
    length = np.linalg.norm(delta)
    facing = [float(np.dot(n, d) / length) for n, d in zip(path_normals, (delta, -delta))]
    hits = path_hits(path, pose, catalog, meshes)
    exact_hits = path_hits(exact_path, pose, catalog, meshes)
    rounded_site_hits, interior_hits = [], []
    for stable_id, face, t in hits:
        endpoint = 0 if t < .5 else 1
        if ([stable_id, face] == [stable_ids[endpoint], triangle_rows[endpoint]]
            and surface_errors[endpoint] <= 1e-7
            and not any(hit[:2] == [stable_id, face] for hit in exact_hits)):
            rounded_site_hits.append(dict(stable_id=stable_id, face=face, segment_fraction=t,
                                          endpoint=endpoint,
                                          axial_distance_m=float(length * abs(t - endpoint)),
                                          source_position_error_m=surface_errors[endpoint]))
        else:
            interior_hits.append([stable_id, face, t])
    return dict(side=attachment["side"], joint=attachment["joint"], length_m=float(length),
                endpoint_facing=facing, max_fixed_site_error_m=max(surface_errors),
                source_bone_hits=hits, barycentric_path_hits=exact_hits,
                rounded_fixed_site_hits=rounded_site_hits, unresolved_source_bone_hits=interior_hits)


def check(launch, anchors, motion, steps):
    argv = json.loads(launch.read_text())["argv"]
    audit.ag.RIGID, audit.ag.BONES = map(Path, (argv[1], argv[3]))
    _, payload, _, _, _, _, _, catalog, _, vertex_offset, index_offset = audit.ag.read_payloads()
    registration = json.loads(anchors.read_text())
    sha = hashlib.sha256(payload).hexdigest()
    assert registration["bone_payload_sha256"] == sha
    meshes = load_local_meshes(payload, catalog, vertex_offset, index_offset)
    needed = {entry["body"] for entry in catalog.values()}
    by_step = {step: {} for step in steps}
    with motion.open() as stream:
        for row in csv.DictReader(stream):
            step, body = int(row["accepted_step"]), int(row["body_index"])
            if step in by_step and body in needed:
                assert body not in by_step[step]
                by_step[step][body] = row_pose(row)
    result = []
    for step, pose in by_step.items():
        assert pose.keys() == needed, (step, sorted(needed - pose.keys()))
        checks = [check_path(a, pose, catalog, meshes) for a in registration["candidates"]]
        result.append(dict(accepted_step=step, shoulders=checks))
    return dict(scope="Bilateral superior glenohumeral material attachments across sampled accepted poses",
                bone_payload_sha256=sha, source_bone_banks=len(meshes),
                captures=result, input_sha256={
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (launch, anchors, motion)
                })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("launch", "anchors", "motion", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    report = check(args.launch, args.anchors, args.motion, args.steps)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for pose in report["captures"]:
        for path in pose["shoulders"]:
            assert max(path["max_fixed_site_error_m"], 0.0) <= 1e-7
            assert min(path["endpoint_facing"]) > .015
            assert not path["unresolved_source_bone_hits"], path
            assert not path["barycentric_path_hits"], path
