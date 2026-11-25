"""Retest accepted fold witnesses on their registered source and uncompressed posed material."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bindings import audit, read_skin
from compile_connected_joint import digest
from fast_skin_intersection_audit import exact_integer_pairs
from rebind_human_skin_reference import decode


def read(path):
    return json.loads(path.read_bytes())


def diagnose(material, case, report):
    registry = read(material / "material-boundary.json")
    skin = material / "registered-boundary.nhskin"
    frame = Path(registry["initial_bodies_path"])
    if digest(frame) != registry["initial_frame_sha256"] or digest(skin) != registry["nhskin_sha256"]:
        raise ValueError("Wrong material source")
    result = read(report)
    path = case / "native-traces/accepted-geometry" / f"step-{result['accepted_step']}.receipt.json"
    receipt = read(path)
    pack = Path(receipt["accepted_pack_path"])
    if digest(pack) != result["accepted_pack_sha256"] or digest(pack) != receipt["pack_file_sha256"]:
        raise ValueError("Unmatched fold witnesses")
    actual, faces = read_skin(pack, receipt)
    registered = decode(skin.read_bytes())
    if not np.array_equal(registered["inds"], faces):
        raise ValueError("Changed material connectivity")
    pairs = np.asarray(result["self_intersections"]["triangle_pairs"], dtype=int).reshape(-1, 2)
    summary = dict(pack=receipt["pack_file_sha256"], accepted_step=result["accepted_step"],
                   witness_pairs=len(pairs), cases={})
    if not len(pairs):
        return summary

    face_ids, local_pairs = np.unique(pairs.reshape(-1), return_inverse=True)
    local_pairs = local_pairs.reshape(-1, 2)
    vertex_ids, local_faces = np.unique(faces[face_ids], return_inverse=True)
    local_faces = local_faces.reshape(-1, 3)
    weights = registered["weights"][vertex_ids]
    source = registered["verts"][vertex_ids, :3].astype(float)
    reference = {b["body_index"]: b["current"] for b in read(frame)["bodies"]}
    current = {b["body_index"]: b for b in receipt["accepted_body_poses"]}
    posed = np.zeros_like(source)
    for k, body_id in enumerate(registered["binds"]["body"]):
        if not weights[:, k].any():
            continue
        ref, now = reference[body_id], current[body_id]
        relative = Rotation.from_quat(now["quaternion_xyzw"]) * Rotation.from_quat(ref["orientation_world_xyzw"]).inv()
        posed += weights[:, k, None] * (relative.apply(source-ref["position_world_m"]) + now["position_m"])
    native = actual[vertex_ids]
    gap = np.linalg.norm(native-posed, axis=1) * 1000.
    for name, coords in (("source", source), ("uncompressed_skinning", posed), ("native", native)):
        records = audit.clearance._exact_surface_records(np.asarray(coords, dtype="f"), local_faces)
        hits = exact_integer_pairs(((records[i], records[j]) for i, j in local_pairs), same_surface=True)
        summary["cases"][name] = [[int(face_ids[i]), int(face_ids[j])] for i, j in hits["triangle_pairs"]]
    if summary["cases"]["native"] != pairs.tolist():
        raise ValueError("Mismatch reproducing native Float32 fold witnesses")
    summary["dominant_donor_by_witness_vertex"] = dict(Counter(map(str, registered["binds"]["body"][weights.argmax(axis=1)])))
    summary["skin_vertices"] = len(vertex_ids)
    summary["max_native_to_uncompressed_mm"] = float(gap.max())
    summary["median_native_to_uncompressed_mm"] = float(np.median(gap))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("material", "case", "report"):
        parser.add_argument(arg, type=Path)
    print(json.dumps(diagnose(**vars(parser.parse_args())), indent=2))
