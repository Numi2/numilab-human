"""Source-bound one-ring refinement for a measured accepted thorax counterexample.

This emits the existing NHANAT1 ABI5 payload and its existing anatomy receipt.
It does not change original source positions.  Selected edge midpoints are
propagated across exact reciprocal diaphragm/lung faces before triangles split.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

from . import resting_anatomy_interface_patch as base_owner

BASE_PAYLOAD = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/diaphragm-cardiac-binding-001/resting-thorax.nhanatomy")
BASE_RECEIPT = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/diaphragm-cardiac-binding-001/resting-anatomy-receipt.json")
RESPIRATION_CONFIG = Path("/Users/n/numi-human-resting-lab-20261005/matter/examples/resting-reference-respiration.json")
BASE_PAYLOAD_SHA256 = "cd64c91eaaa768e809b3b9f22cff6551de3cb14f5dc63ac56a9d794f755b0f52"
BASE_RECEIPT_SHA256 = "c09cc9cc0a6a3449abd954725c11a65f21cb8bcfbf35adc946d1d65c67c38096"
WITNESS_PATH = Path("/Users/n/numi-human-resting-evidence-20261005/diaphragm-common-map-native-001/accepted-geometry/thorax-reciprocal-witness-305-311-286-5654-002.json")
WITNESS_SHA256 = "7485571649040c972f384794a41a30ebc6fa8e9ecbba41626e9a00fa534a92e7"
HEADER = base_owner.HEADER
RECORD = base_owner.RECORD


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coord_key(vertices6: np.ndarray, vertex: int):
    return tuple(float(x) for x in vertices6[int(vertex), :3])


def edge_key(a: int, b: int):
    return (min(int(a), int(b)), max(int(a), int(b)))


def edge_incidence(faces: np.ndarray):
    incidence = defaultdict(list)
    for fi, triangle in enumerate(np.asarray(faces, dtype=np.int64)):
        a, b, c = map(int, triangle)
        for x, y in ((a, b), (b, c), (c, a)):
            incidence[edge_key(x, y)].append(int(fi))
    return incidence


def _split_triangle(vertices6: np.ndarray, triangle, split_edges, midpoint_ids):
    a, b, c = map(int, triangle)
    edges = (edge_key(a, b), edge_key(b, c), edge_key(c, a))
    count = sum(edge in split_edges for edge in edges)
    if count == 0:
        return [(a, b, c)]
    if count == 3:
        mab, mbc, mca = (midpoint_ids[e] for e in edges)
        return [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
    boundary = []
    for i, vertex in enumerate((a, b, c)):
        boundary.append(vertex)
        edge = edges[i]
        if edge in split_edges:
            boundary.append(midpoint_ids[edge])
    # Ear-clip the convex boundary polygon. A plain fan can choose a split
    # midpoint as its pivot, producing a zero-area child along the split edge.
    # Select ears by the exact source-space middle point so an opposite-wound
    # reciprocal face gets the same geometric diagonal.
    old = vertices6[np.asarray((a, b, c)), :3].astype(np.float64)
    reference_normal = np.cross(old[1] - old[0], old[2] - old[0])
    polygon = list(boundary)
    result = []
    while len(polygon) > 3:
        ears = []
        for i, curr in enumerate(polygon):
            prev = polygon[(i - 1) % len(polygon)]
            nxt = polygon[(i + 1) % len(polygon)]
            tri = vertices6[np.asarray((prev, curr, nxt)), :3].astype(np.float64)
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if float(np.dot(reference_normal, normal)) > 1e-30:
                # Avoid an ear that leaves only the three collinear points on
                # one split edge as the final polygon.
                if len(polygon) == 4:
                    remaining = [polygon[j] for j in range(4) if j != i]
                    last = vertices6[np.asarray(remaining), :3].astype(np.float64)
                    last_normal = np.cross(last[1] - last[0], last[2] - last[0])
                    if float(np.dot(reference_normal, last_normal)) <= 1e-30:
                        continue
                ears.append((coord_key(vertices6, curr), i, (prev, curr, nxt)))
        if not ears:
            raise ValueError("could not find a positive-area conforming refinement ear")
        _, index, triangle = min(ears)
        result.append(triangle)
        polygon.pop(index)
    result.append(tuple(polygon))
    return result


def refine_surface_edges(vertices6, faces, split_edges):
    vertices6 = np.asarray(vertices6, dtype=np.float32).copy()
    faces = np.asarray(faces, dtype=np.int64).copy()
    split_edges = {edge_key(*edge) for edge in split_edges}
    if not split_edges:
        identity = {i: [i] for i in range(len(faces))}
        return vertices6, faces, identity, {"split_edge_count": 0, "affected_face_count": 0, "added_vertex_count": 0, "added_face_count": 0}
    incidence = edge_incidence(faces)
    for edge in split_edges:
        adjacent = incidence.get(edge, [])
        if len(adjacent) != 2:
            raise ValueError(f"refinement edge {edge} is not a two-face manifold edge: {adjacent}")
    midpoint_ids = {}
    new_vertices = []
    for edge in sorted(split_edges):
        a, b = edge
        point = ((vertices6[a, :3].astype(np.float64) + vertices6[b, :3].astype(np.float64)) * .5).astype(np.float32)
        normal = vertices6[a, 3:6].astype(np.float64) + vertices6[b, 3:6].astype(np.float64)
        length = float(np.linalg.norm(normal))
        if not np.isfinite(point).all() or not math.isfinite(length) or length <= 1e-12:
            raise ValueError(f"cannot form finite source midpoint for edge {edge}")
        row = np.concatenate([point, (normal / length).astype(np.float32)])
        midpoint_ids[edge] = len(vertices6) + len(new_vertices)
        new_vertices.append(row)
    if new_vertices:
        combined_vertices = np.vstack([vertices6, np.asarray(new_vertices, dtype=np.float32)])
    else:
        combined_vertices = vertices6
    affected = {fi for edge in split_edges for fi in incidence[edge]}
    old_to_new = {}
    refined_faces = []
    for fi, triangle in enumerate(faces):
        children = _split_triangle(combined_vertices, triangle, split_edges, midpoint_ids) if fi in affected else [tuple(map(int, triangle))]
        start = len(refined_faces)
        for child in children:
            tri = combined_vertices[np.asarray(child), :3].astype(np.float64)
            if float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))) <= 1e-15:
                raise ValueError(f"refinement made a zero-area child face from {fi}")
            old_tri = vertices6[np.asarray(triangle), :3].astype(np.float64)
            old_n = np.cross(old_tri[1] - old_tri[0], old_tri[2] - old_tri[0])
            new_n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if float(np.dot(old_n, new_n)) <= 0:
                raise ValueError(f"refinement reversed child-face winding from {fi}")
            refined_faces.append(tuple(map(int, child)))
        old_to_new[fi] = list(range(start, len(refined_faces)))
    result_faces = np.asarray(refined_faces, dtype=np.int64)
    return combined_vertices, result_faces, old_to_new, {
        "split_edge_count": len(split_edges), "affected_face_count": len(affected),
        "added_vertex_count": len(new_vertices), "added_face_count": len(result_faces) - len(faces),
        "split_edges": [list(edge) for edge in sorted(split_edges)],
        "affected_face_indices": sorted(affected),
    }


def _patch_face_ids(row, key):
    if key == "registered_lung_face_index_ranges":
        return {index for start, end in row[key] for index in range(int(start), int(end))}
    start, count = int(row["diaphragm_patch_face_start"]), int(row["diaphragm_patch_face_count"])
    return set(range(start, start + count))


def interface_face_mates(rows, receipt):
    diaphragm = rows[311]
    maps = {}
    reverse = {}
    for row in receipt["provenance"]["diaphragm_lung_interface"]["interface_rows"]:
        sid = int(row["lung_stable_id"])
        lung = rows[sid]
        lobe_ids = _patch_face_ids(row, "registered_lung_face_index_ranges")
        diaphragm_ids = _patch_face_ids(row, "diaphragm_patch_face_start")
        if len(lobe_ids) != len(diaphragm_ids):
            raise ValueError(f"source reciprocal patch face counts differ for lobe {sid}")
        def key(surface, face):
            return tuple(sorted(coord_key(surface["vertices6"], v) for v in surface["faces"][face]))
        lobe_by_key = {}
        for fi in lobe_ids:
            k = key(lung, fi)
            if k in lobe_by_key:
                raise ValueError(f"duplicate exact lobe patch face in stable ID {sid}")
            lobe_by_key[k] = fi
        diaphragm_by_key = {}
        for fi in diaphragm_ids:
            k = key(diaphragm, fi)
            if k in diaphragm_by_key:
                raise ValueError("duplicate exact diaphragm patch face")
            diaphragm_by_key[k] = fi
        if set(lobe_by_key) != set(diaphragm_by_key):
            raise ValueError(f"source reciprocal patch does not match exactly for lobe {sid}")
        for k, lobe_fi in lobe_by_key.items():
            diaphragm_fi = diaphragm_by_key[k]
            maps[(sid, lobe_fi)] = (311, diaphragm_fi)
            reverse[(311, diaphragm_fi)] = (sid, lobe_fi)
    return maps, reverse


def propagate_interface_edges(rows, receipt, seeds):
    split_edges = {sid: {edge_key(*edge) for edge in edges} for sid, edges in seeds.items()}
    forward, reverse = interface_face_mates(rows, receipt)
    mates = dict(forward)
    mates.update(reverse)
    incidence = {sid: edge_incidence(rows[sid]["faces"]) for sid in split_edges}
    queue = deque((sid, edge) for sid, edges in split_edges.items() for edge in edges)
    while queue:
        sid, edge = queue.popleft()
        for face_id in incidence[sid].get(edge, []):
            mate = mates.get((sid, face_id))
            if mate is None:
                continue
            other_sid, other_face = mate
            other = rows[other_sid]
            endpoints = {coord_key(rows[sid]["vertices6"], v) for v in edge}
            candidates = []
            tri = other["faces"][other_face]
            for i in range(3):
                e = edge_key(int(tri[i]), int(tri[(i + 1) % 3]))
                if {coord_key(other["vertices6"], v) for v in e} == endpoints:
                    candidates.append(e)
            if len(candidates) != 1:
                raise ValueError(f"could not map refined reciprocal edge {(sid, edge)} to mate {(other_sid, other_face)}")
            other_edge = candidates[0]
            if other_edge not in split_edges.setdefault(other_sid, set()):
                if other_sid not in incidence:
                    incidence[other_sid] = edge_incidence(other["faces"])
                split_edges[other_sid].add(other_edge)
                queue.append((other_sid, other_edge))
    return split_edges


def remap_ids(ids, old_to_new):
    return sorted(new for old in sorted(ids) for new in old_to_new[int(old)])


def face_set_boundary_edges(faces, selected_face_ids):
    incidence = edge_incidence(faces)
    selected_face_ids = set(selected_face_ids)
    boundary = []
    for edge, adjacent in incidence.items():
        inside = sum(face in selected_face_ids for face in adjacent)
        if inside == 1 and len(adjacent) == 2:
            boundary.append(edge)
        elif inside not in (0, len(adjacent)):
            raise ValueError("interface patch has a branched or nonmanifold boundary edge")
    return sorted(boundary)


def _area(vertices6, faces, selected):
    tri = vertices6[np.asarray(faces, dtype=np.int64)[np.asarray(selected, dtype=np.int64)], :3].astype(np.float64)
    return float(.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def _surface_identity_sha256(sid, row):
    digest = hashlib.sha256()
    digest.update(struct.pack("<IIII", int(sid), int(row["body_index"]), int(row["layer"]), int(row["flags"])))
    digest.update(np.asarray(row["vertices6"], dtype="<f4").tobytes())
    digest.update(np.asarray(row["faces"], dtype="<u4").tobytes())
    return digest.hexdigest()


def _pack(header, rows):
    surface_records, vertex_blocks, index_blocks = [], [], []
    vertex_count = index_count = 0
    for sid in sorted(rows):
        row = rows[sid]
        vertices6 = np.asarray(row["vertices6"], dtype="<f4")
        faces = np.asarray(row["faces"], dtype=np.uint32)
        if not np.isfinite(vertices6).all() or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices6))):
            raise ValueError(f"invalid refined NHANAT surface {sid}")
        surface_records.append(RECORD.pack(int(row["body_index"]), vertex_count, len(vertices6), index_count, faces.size, sid, int(row["layer"]), int(row["flags"])))
        vertex_blocks.append(vertices6.tobytes())
        index_blocks.append((faces.ravel() + vertex_count).astype("<u4").tobytes())
        vertex_count += len(vertices6)
        index_count += faces.size
    data = HEADER.pack(header[0], header[1], len(rows), vertex_count, index_count, header[5], header[6])
    data += b"".join(surface_records) + b"".join(vertex_blocks) + b"".join(index_blocks)
    return data, vertex_count, index_count


def build_candidate(base_payload: Path, base_receipt: Path, witness: Path, output_dir: Path):
    for path, expected in ((base_payload, BASE_PAYLOAD_SHA256), (base_receipt, BASE_RECEIPT_SHA256), (witness, WITNESS_SHA256)):
        if sha(path) != expected:
            raise ValueError(f"pinned input SHA mismatch: {path}")
    witness_report = json.loads(witness.read_text())
    if witness_report.get("pair", {}).get("lung_triangle_index") != 286 or witness_report.get("pair", {}).get("diaphragm_triangle_index") != 5654:
        raise ValueError("witness report does not bind the intended selected triangle pair")
    header, rows = base_owner.parse_payload(base_payload)
    receipt = json.loads(base_receipt.read_text())
    if receipt.get("payload", {}).get("sha256") != BASE_PAYLOAD_SHA256 or receipt.get("payload", {}).get("surface_count") != len(receipt.get("provenance", {}).get("source_id_map", {})):
        raise ValueError("base anatomy receipt identity mismatch")
    if set(rows) != {int(sid) for sid in receipt["provenance"]["source_id_map"]}:
        raise ValueError("base receipt stable-ID set differs from NHANAT records")
    cardiac_binding = receipt["provenance"]["cardiac_geometry_binding"]
    cardiac_stable_ids = sorted(
        {int(row["stable_id"]) for row in cardiac_binding["passive_outer_heart_bindings"]}
        | {int(row["stable_id"]) for row in cardiac_binding["source_member_bindings"]}
    )
    source_cardiac_identity = {sid: _surface_identity_sha256(sid, rows[sid]) for sid in cardiac_stable_ids}
    original_volumes = {sid: abs(base_owner.signed_volume(rows[sid]["vertices6"][:, :3].astype(np.float64), rows[sid]["faces"])) for sid in (305, 311)}
    original_rows = {sid: {"vertices6": rows[sid]["vertices6"].copy(), "faces": rows[sid]["faces"].copy()} for sid in (305, 311)}
    seed_edges = {}
    for sid, face_id in ((305, 286), (311, 5654)):
        tri = rows[sid]["faces"][face_id]
        seed_edges[sid] = [edge_key(int(tri[i]), int(tri[(i + 1) % 3])) for i in range(3)]
    split_edges = propagate_interface_edges(rows, receipt, seed_edges)
    refinement = {}
    old_to_new = {}
    for sid, edges in split_edges.items():
        vertices6, faces, mapping, result = refine_surface_edges(rows[sid]["vertices6"], rows[sid]["faces"], edges)
        rows[sid]["vertices6"] = vertices6
        rows[sid]["faces"] = faces
        old_to_new[sid] = mapping
        volume = abs(base_owner.signed_volume(vertices6[:, :3].astype(np.float64), faces))
        delta = volume - original_volumes[sid]
        if abs(delta) > 1e-12:
            raise ValueError(f"refinement changed closed source volume for {sid} by {delta:.9g} m3")
        topo = base_owner.topology_report(faces)
        if topo["nonmanifold_edge_count"] or topo["orientation_error_edge_count"] or topo["boundary_branch_vertex_count"]:
            raise ValueError(f"refinement invalidated source topology for {sid}: {topo}")
        if sid in (305,):
            if topo["boundary_edge_count"] != 0 or topo["boundary_loop_count"] != 0:
                raise ValueError("lung envelope refinement opened a closed lobe")
        if sid == 311 and (topo["boundary_edge_count"] != 10 or topo["boundary_loop_count"] != 1):
            raise ValueError("diaphragm refinement changed the retained ten-edge source aperture")
        target_faces = {305: (284, 286), 311: (33291, 5654)}[sid]
        result.update({
            "source_volume_m3_before": original_volumes[sid],
            "source_volume_m3_after": volume,
            "source_volume_delta_m3": delta,
            "topology_after": {k:v for k,v in topo.items() if k != "boundary_edges"},
            "source_face_to_candidate_face_indices": {str(fi): mapping[fi] for fi in target_faces},
        })
        refinement[str(sid)] = result
    # Identity mappings make receipt remapping explicit for the untouched lungs.
    for sid in (305, 306, 307, 308, 309, 311):
        old_to_new.setdefault(sid, {i:[i] for i in range(len(rows[sid]["faces"]))})
    interface = receipt["provenance"]["diaphragm_lung_interface"]
    old_interface_rows = interface["interface_rows"]
    updated_interface_rows = []
    for original in old_interface_rows:
        row = dict(original)
        sid = int(row["lung_stable_id"])
        old_lung_ids = _patch_face_ids(original, "registered_lung_face_index_ranges")
        new_lung_ids = remap_ids(old_lung_ids, old_to_new[sid])
        old_start, old_count = int(row["diaphragm_patch_face_start"]), int(row["diaphragm_patch_face_count"])
        new_d_ids = remap_ids(range(old_start, old_start + old_count), old_to_new[311])
        if len(new_d_ids) != len(new_lung_ids) or new_d_ids != list(range(new_d_ids[0], new_d_ids[0] + len(new_d_ids))):
            raise ValueError(f"refined reciprocal interface faces are not a matched contiguous diaphragm patch for lobe {sid}: lung={len(new_lung_ids)} diaphragm={len(new_d_ids)} d_range={new_d_ids[:4]}..{new_d_ids[-4:]}")
        row["registered_lung_face_index_ranges"] = base_owner.index_ranges(new_lung_ids)
        row["registered_lung_face_count"] = len(new_lung_ids)
        row["diaphragm_patch_face_start"] = new_d_ids[0]
        row["diaphragm_patch_face_count"] = len(new_d_ids)
        row["patch_area_m2"] = _area(rows[sid]["vertices6"], rows[sid]["faces"], new_lung_ids)
        boundary = face_set_boundary_edges(rows[sid]["faces"], new_lung_ids)
        row["shared_boundary_edge_count"] = len(boundary)
        row["shared_boundary_vertex_count"] = len({v for edge in boundary for v in edge})
        updated_interface_rows.append(row)
    interface["interface_rows"] = updated_interface_rows
    # Verify reciprocal triangles remain exact, opposite-winding interface cells.
    reciprocal_check = {}
    for row in updated_interface_rows:
        sid = int(row["lung_stable_id"])
        lobe_ids = _patch_face_ids(row, "registered_lung_face_index_ranges")
        d_ids = _patch_face_ids(row, "diaphragm_patch_face_start")
        def tri_map(surface, face_ids):
            out = {}
            for fi in face_ids:
                tri_ids = surface["faces"][fi]
                key = tuple(sorted(coord_key(surface["vertices6"], v) for v in tri_ids))
                oriented = tuple(coord_key(surface["vertices6"], v) for v in tri_ids)
                if key in out: raise ValueError("duplicate triangle in refined reciprocal patch")
                out[key] = oriented
            return out
        lt = tri_map(rows[sid], lobe_ids); dt = tri_map(rows[311], d_ids)
        if set(lt) != set(dt): raise ValueError(f"refined reciprocal patch geometry differs for lobe {sid}")
        opposite = 0
        for key in lt:
            a, b = lt[key], dt[key]
            if a != (b[0],b[2],b[1]) and a != (b[1],b[0],b[2]) and a != (b[2],b[1],b[0]):
                # Require opposite orientation by signed normal dot product,
                # tolerating only cyclic starting-point changes.
                v=np.asarray(a,dtype=np.float64);w=np.asarray(b,dtype=np.float64)
                na=np.cross(v[1]-v[0],v[2]-v[0]);nb=np.cross(w[1]-w[0],w[2]-w[0])
                if float(np.dot(na,nb))>=0: raise ValueError("refined interface winding is not reciprocal")
            else:
                opposite+=1
        reciprocal_check[str(sid)]={"matched_face_count":len(lt),"opposite_winding_face_count":len(lt)}
    topology = base_owner.topology_report(rows[311]["faces"])
    topology.pop("boundary_edges", None)
    interface["diaphragm_topology_after_patch"] = topology
    interface["added_reversed_interface_face_count"] = sum(row["diaphragm_patch_face_count"] for row in updated_interface_rows)
    interface["added_reversed_interface_surface_area_m2"] = sum(row["patch_area_m2"] for row in updated_interface_rows)
    interface["conforming_refinement"] = {"algorithm":"source-edge-midpoint-one-ring-v1","split_edges_by_stable_id":{str(sid):[list(edge) for edge in sorted(edges)] for sid,edges in split_edges.items()},"surface_refinement":refinement,"reciprocal_patch_check":reciprocal_check,"original_source_positions_modified":False,"qualification":"source topology and reciprocal interface preserved after local refinement; accepted-state dynamic regression required"}
    areas = base_owner.source_area_basis(rows)
    respiratory = receipt["functional_bindings"]["respiratory_geometry_binding"]
    previous_area = respiratory["diaphragm_effective_area_m2"]
    respiratory.update({"diaphragm_effective_area_m2":areas["effective_area_m2"],"per_lobe_effective_area_m2":areas["per_lobe_effective_area_m2"],"finite_displacement_sensitivity_m2":areas["finite_displacement_sensitivity_m2"],"calibration_displacement_m":areas["calibration_displacement_m"],"source_refinement_area_update":{"previous_area_m2":previous_area,"refined_area_m2":areas["effective_area_m2"],"relative_change":(areas["effective_area_m2"]-previous_area)/previous_area,"basis":"recomputed from closed refined NHANAT lobe triangles using the unchanged accepted respiratory basis"}})
    for sid in (305,306,307,308,309):
        receipt["thorax_source_volume_m3"]["five_lung_envelopes"][sid-305]=abs(base_owner.signed_volume(rows[sid]["vertices6"][:,:3].astype(np.float64),rows[sid]["faces"]))
    receipt["thorax_source_volume_m3"]["sum"]=sum(receipt["thorax_source_volume_m3"]["five_lung_envelopes"])
    for sid in (305,311):
        entry=receipt["provenance"]["source_id_map"][str(sid)]
        entry.setdefault("repair",{})["accepted_cycle_local_refinement"]={"method":"one-ring conforming edge-midpoint subdivision with reciprocal interface propagation","source_mesh_face_indices":{"305":286,"311":5654},"original_source_positions_modified":False,"source_volume_delta_m3":refinement[str(sid)]["source_volume_delta_m3"] if str(sid) in refinement else 0.0,"status":"candidate; accepted-state four-frame geometric regression pending"}
    receipt["qualification"]["diaphragm_lung_interface"]="locally refined reciprocal interface candidate; source topology, volume and static patch checks passed; native accepted-state four-frame regression pending"
    receipt["qualification"]["inter_lobe_intersection"]="not reassessed by the localized diaphragm/lung refinement candidate"
    candidate_cardiac_identity = {sid: _surface_identity_sha256(sid, rows[sid]) for sid in cardiac_stable_ids}
    if source_cardiac_identity != candidate_cardiac_identity:
        raise ValueError("localized thoracic refinement changed a cardiac source surface record")
    payload_bytes, vertex_count, index_count = _pack(header, rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    payload_path=output_dir/"resting-thorax.nhanatomy"; payload_path.write_bytes(payload_bytes)
    payload_hash=sha(payload_path)
    cardiac_binding["output_anatomy_payload_sha256"] = payload_hash
    cardiac_binding["retained_geometry"] = (
        "The cardiac binding step preserved non-RA/RV source records as recorded. "
        "This descendant adds local edge-midpoint refinement only to lung ID 305 and diaphragm ID 311; "
        "all exact surface arrays for the cardiac IDs listed in downstream_anatomy_refinement are byte-identical."
    )
    cardiac_binding["downstream_anatomy_refinement"] = {
        "source_payload_sha256": BASE_PAYLOAD_SHA256,
        "output_payload_sha256": payload_hash,
        "modified_stable_ids": [305, 311],
        "preserved_cardiac_surface_sha256": {str(sid): candidate_cardiac_identity[sid] for sid in cardiac_stable_ids},
        "preserved_cardiac_subset_byte_identity": True,
    }
    receipt["payload"].update({"path":str(payload_path),"sha256":payload_hash,"surface_count":len(rows),"vertex_count":vertex_count,"index_count":index_count,"registration_fingerprint32":f"{header[5]:08x}"})
    receipt["functional_bindings"]["anatomy_payload_sha256"]=payload_hash
    receipt["provenance"]["conforming_local_refinement"]={"source_payload_path":str(base_payload),"source_payload_sha256":BASE_PAYLOAD_SHA256,"source_receipt_path":str(base_receipt),"source_receipt_sha256":BASE_RECEIPT_SHA256,"accepted_witness_path":str(witness),"accepted_witness_sha256":WITNESS_SHA256,"algorithm":"one-ring conforming edge-midpoint subdivision with exact reciprocal-face propagation","targets":[{"stable_id":305,"face_index":286},{"stable_id":311,"face_index":5654}],"refinement":refinement,"propagated_split_edges_by_stable_id":{str(sid):[list(edge) for edge in sorted(edges)] for sid,edges in split_edges.items()},"source_vertex_coordinates_modified":False,"source_closed_volume_preserved_within_m3":1e-12,"respiratory_geometry_binding_area_before_m2":previous_area,"respiratory_geometry_binding_area_after_m2":areas["effective_area_m2"],"accepted_native_cycle_qualification":"pending"}
    receipt_path=output_dir/"resting-anatomy-receipt.json"; receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    if RESPIRATION_CONFIG.exists():
        config=json.loads(RESPIRATION_CONFIG.read_text()); config["diaphragm_area_m2"]=areas["effective_area_m2"]
        config.setdefault("parameter_scope",{})["diaphragm_area_refinement_note"]="Updated from the exact refined closed-lobe geometry basis; inferred reference parameter, not measured subject data."
        (output_dir/"resting-reference-respiration.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
    result={"candidate_payload_path":str(payload_path),"candidate_payload_sha256":payload_hash,"candidate_receipt_path":str(receipt_path),"candidate_receipt_sha256":sha(receipt_path),"candidate_config_path":str(output_dir/"resting-reference-respiration.json"),"candidate_config_sha256":sha(output_dir/"resting-reference-respiration.json") if (output_dir/"resting-reference-respiration.json").exists() else None,"vertex_count":vertex_count,"index_count":index_count,"refinement":refinement,"respiratory_area_before_m2":previous_area,"respiratory_area_after_m2":areas["effective_area_m2"],"reciprocal_patch_check":reciprocal_check}
    (output_dir/"refinement-result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description="Build a source-bound NHANAT conforming refinement candidate")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-cycle-refinement",action="store_true",
        help="seed one bounded conforming batch from all 31 pinned peak-phase outside-patch positive segments")
    args=parser.parse_args(argv)
    if args.batch_cycle_refinement:
        result=build_batch_candidate(args.output)
    else:
        result=build_candidate(BASE_PAYLOAD,BASE_RECEIPT,WITNESS_PATH,args.output)
    print(json.dumps(result,sort_keys=True))


BATCH_BASE_PAYLOAD = Path("/Users/n/numi-human-resting-evidence-20261005/thorax-conforming-refinement-004/resting-thorax.nhanatomy")
BATCH_BASE_RECEIPT = Path("/Users/n/numi-human-resting-evidence-20261005/thorax-conforming-refinement-004/resting-anatomy-receipt.json")
BATCH_PEAK_WITNESS = Path("/Users/n/numi-human-resting-evidence-20261005/thorax-conforming-native-001/accepted-geometry/full-outside-patch-305-311-step-639.json")
BATCH_BASELINE_COMPARISON = Path("/Users/n/numi-human-resting-evidence-20261005/thorax-conforming-native-001/accepted-geometry/outside-patch-segment-baseline-comparison-001.json")
BATCH_CONFIG = Path("/Users/n/numi-human-resting-evidence-20261005/thorax-conforming-refinement-004/resting-reference-respiration.json")
BATCH_BASE_PAYLOAD_SHA256 = "8e6d230cc25a5d322aa10d5f40c676e9b73654de5e6e48b067c5f6739f8b2dd6"
BATCH_BASE_RECEIPT_SHA256 = "74345dc5340d78717b60e6ec9001b6f2f8eebc0df5af862faf085c155a5794b5"
BATCH_PEAK_WITNESS_SHA256 = "be12043904f074b5b002aab1e3a36a950be39e4693f416b0c9fa0f1a784e37f0"
BATCH_BASELINE_COMPARISON_SHA256 = "e125dab5aa3d9625f6aa31efd7a81ed849a32c024570e14247ce14a17ba8ef40"
BATCH_CONFIG_SHA256 = "1da773df52785e57fd67b05f8c9d71f5c075d4855c126ae6c2c89ffc745f88c0"



def build_batch_candidate(output_dir: Path):
    inputs = (
        (BATCH_BASE_PAYLOAD, BATCH_BASE_PAYLOAD_SHA256),
        (BATCH_BASE_RECEIPT, BATCH_BASE_RECEIPT_SHA256),
        (BATCH_PEAK_WITNESS, BATCH_PEAK_WITNESS_SHA256),
        (BATCH_BASELINE_COMPARISON, BATCH_BASELINE_COMPARISON_SHA256),
        (BATCH_CONFIG, BATCH_CONFIG_SHA256),
    )
    for path, expected in inputs:
        if sha(path) != expected:
            raise ValueError(f"pinned batch input SHA mismatch: {path}")
    peak = json.loads(BATCH_PEAK_WITNESS.read_text())
    comparison = json.loads(BATCH_BASELINE_COMPARISON.read_text())
    if peak.get("step") != 639 or comparison.get("peak_outside_patch_positive_segment_pair_count") != 31:
        raise ValueError("batch evidence is not the retained 31-pair peak-step comparison")
    if comparison.get("step639_audit_sha256") != BATCH_PEAK_WITNESS_SHA256:
        raise ValueError("baseline comparison does not bind the pinned peak geometry audit")
    peak_witnesses = {
        (int(row["lung_face"]), int(row["diaphragm_face"])): row
        for row in peak.get("witnesses", [])
    }
    selected = []
    for row in comparison.get("pairs", []):
        key = (int(row["lung_face"]), int(row["diaphragm_face"]))
        hit = peak_witnesses.get(key)
        if row.get("baseline_class") == "positive_segment":
            raise ValueError(f"selected batch pair already crossed at step0: {key}")
        if hit is None or hit.get("intersection_point_count_unique", 0) < 2 or hit.get("shared_vertex_count", 0) >= 2:
            raise ValueError(f"selected batch pair is not a retained positive segment at step639: {key}")
        selected.append(key)
    if len(selected) != 31 or len(set(selected)) != 31:
        raise ValueError("batch face-pair selection must contain 31 unique phase-induced segments")

    header, rows = base_owner.parse_payload(BATCH_BASE_PAYLOAD)
    receipt = json.loads(BATCH_BASE_RECEIPT.read_text())
    if receipt.get("payload", {}).get("sha256") != BATCH_BASE_PAYLOAD_SHA256:
        raise ValueError("batch base receipt anatomy identity mismatch")
    if set(rows) != {int(sid) for sid in receipt["provenance"]["source_id_map"]}:
        raise ValueError("batch base receipt stable-ID set differs from NHANAT records")
    cardiac_binding = receipt["provenance"]["cardiac_geometry_binding"]
    cardiac_ids = sorted(
        {int(row["stable_id"]) for row in cardiac_binding["passive_outer_heart_bindings"]}
        | {int(row["stable_id"]) for row in cardiac_binding["source_member_bindings"]}
    )
    cardiac_identity_before = {sid: _surface_identity_sha256(sid, rows[sid]) for sid in cardiac_ids}

    targets = {
        305: {int(a) for a, _ in selected},
        311: {int(b) for _, b in selected},
    }
    seed_edges = {sid: set() for sid in targets}
    for sid, face_ids in targets.items():
        for face_id in face_ids:
            if face_id < 0 or face_id >= len(rows[sid]["faces"]):
                raise ValueError(f"witness face {sid}:{face_id} is outside the pinned NHANAT surface")
            tri = rows[sid]["faces"][face_id]
            for i in range(3):
                seed_edges[sid].add(edge_key(int(tri[i]), int(tri[(i + 1) % 3])))
    split_edges = propagate_interface_edges(rows, receipt, seed_edges)
    if not set(split_edges).issubset({305, 306, 307, 308, 309, 311}):
        raise ValueError(f"batch refinement escaped the lung/diaphragm surfaces: {sorted(split_edges)}")

    original_volumes = {
        sid: abs(base_owner.signed_volume(rows[sid]["vertices6"][:, :3].astype(np.float64), rows[sid]["faces"]))
        for sid in split_edges
    }
    old_to_new = {}
    refinement = {}
    for sid, edges in sorted(split_edges.items()):
        before_topology = base_owner.topology_report(rows[sid]["faces"])
        boundary_edges_before = before_topology["boundary_edge_count"]
        vertices6, faces, mapping, detail = refine_surface_edges(rows[sid]["vertices6"], rows[sid]["faces"], edges)
        rows[sid]["vertices6"] = vertices6
        rows[sid]["faces"] = faces
        old_to_new[sid] = mapping
        volume = abs(base_owner.signed_volume(vertices6[:, :3].astype(np.float64), faces))
        delta = volume - original_volumes[sid]
        if abs(delta) > 1e-12:
            raise ValueError(f"batch refinement changed source volume for {sid} by {delta:.9g} m3")
        topo = base_owner.topology_report(faces)
        if topo["nonmanifold_edge_count"] or topo["orientation_error_edge_count"] or topo["boundary_branch_vertex_count"]:
            raise ValueError(f"batch refinement invalidated topology for {sid}: {topo}")
        if sid in (305, 306, 307, 308, 309):
            if topo["boundary_edge_count"] != 0 or topo["boundary_loop_count"] != 0:
                raise ValueError(f"batch refinement opened closed lobe {sid}")
        if sid == 311 and (
            topo["boundary_edge_count"] != boundary_edges_before or
            topo["boundary_loop_count"] != before_topology["boundary_loop_count"]
        ):
            raise ValueError("batch refinement changed the original diaphragm aperture boundary")
        result = dict(detail)
        result.update({
            "source_volume_m3_before": original_volumes[sid],
            "source_volume_m3_after": volume,
            "source_volume_delta_m3": delta,
            "boundary_edge_count_before": boundary_edges_before,
            "boundary_edge_count_after": topo["boundary_edge_count"],
            "topology_after": {k: v for k, v in topo.items() if k != "boundary_edges"},
        })
        if sid in targets:
            result["source_face_to_candidate_face_indices"] = {
                str(fi): mapping[fi] for fi in sorted(targets[sid])
            }
        refinement[str(sid)] = result

    interface = receipt["provenance"]["diaphragm_lung_interface"]
    updated_interface_rows = []
    for original in interface["interface_rows"]:
        row = dict(original)
        sid = int(row["lung_stable_id"])
        old_lung_ids = _patch_face_ids(original, "registered_lung_face_index_ranges")
        lung_map = old_to_new.get(sid, {i: [i] for i in range(len(rows[sid]["faces"]))})
        d_start = int(row["diaphragm_patch_face_start"])
        d_count = int(row["diaphragm_patch_face_count"])
        diaphragm_map = old_to_new.get(311, {i: [i] for i in range(len(rows[311]["faces"]))})
        new_lung_ids = remap_ids(old_lung_ids, lung_map)
        new_d_ids = remap_ids(range(d_start, d_start + d_count), diaphragm_map)
        if len(new_d_ids) != len(new_lung_ids) or not new_d_ids or new_d_ids != list(range(new_d_ids[0], new_d_ids[0] + len(new_d_ids))):
            raise ValueError(f"batch refinement made a noncontiguous or unmatched diaphragm patch for lobe {sid}")
        row["registered_lung_face_index_ranges"] = base_owner.index_ranges(new_lung_ids)
        row["registered_lung_face_count"] = len(new_lung_ids)
        row["diaphragm_patch_face_start"] = new_d_ids[0]
        row["diaphragm_patch_face_count"] = len(new_d_ids)
        row["patch_area_m2"] = _area(rows[sid]["vertices6"], rows[sid]["faces"], new_lung_ids)
        boundary = face_set_boundary_edges(rows[sid]["faces"], new_lung_ids)
        row["shared_boundary_edge_count"] = len(boundary)
        row["shared_boundary_vertex_count"] = len({v for edge in boundary for v in edge})
        updated_interface_rows.append(row)
    interface["interface_rows"] = updated_interface_rows

    reciprocal_check = {}
    for row in updated_interface_rows:
        sid = int(row["lung_stable_id"])
        lobe_ids = _patch_face_ids(row, "registered_lung_face_index_ranges")
        d_start = int(row["diaphragm_patch_face_start"])
        d_ids = set(range(d_start, d_start + int(row["diaphragm_patch_face_count"])))
        def tri_map(surface, face_ids):
            out = {}
            for fi in face_ids:
                tri_ids = surface["faces"][fi]
                key = tuple(sorted(coord_key(surface["vertices6"], v) for v in tri_ids))
                oriented = tuple(coord_key(surface["vertices6"], v) for v in tri_ids)
                if key in out:
                    raise ValueError("duplicate triangle in batch reciprocal patch")
                out[key] = oriented
            return out
        lt = tri_map(rows[sid], lobe_ids)
        dt = tri_map(rows[311], d_ids)
        if set(lt) != set(dt):
            raise ValueError(f"batch reciprocal patch geometry differs for lobe {sid}")
        for key in lt:
            a, b = lt[key], dt[key]
            v = np.asarray(a, dtype=np.float64); w = np.asarray(b, dtype=np.float64)
            na = np.cross(v[1] - v[0], v[2] - v[0])
            nb = np.cross(w[1] - w[0], w[2] - w[0])
            if float(np.dot(na, nb)) >= 0:
                raise ValueError(f"batch reciprocal patch winding is not opposite for lobe {sid}")
        reciprocal_check[str(sid)] = {"matched_face_count": len(lt), "opposite_winding_face_count": len(lt)}

    topo_d = base_owner.topology_report(rows[311]["faces"])
    topo_d.pop("boundary_edges", None)
    interface["diaphragm_topology_after_patch"] = topo_d
    interface["added_reversed_interface_face_count"] = sum(row["diaphragm_patch_face_count"] for row in updated_interface_rows)
    interface["added_reversed_interface_surface_area_m2"] = sum(row["patch_area_m2"] for row in updated_interface_rows)
    area_before = receipt["functional_bindings"]["respiratory_geometry_binding"]["diaphragm_effective_area_m2"]
    areas = base_owner.source_area_basis(rows)
    respiratory = receipt["functional_bindings"]["respiratory_geometry_binding"]
    respiratory.update({
        "diaphragm_effective_area_m2": areas["effective_area_m2"],
        "per_lobe_effective_area_m2": areas["per_lobe_effective_area_m2"],
        "finite_displacement_sensitivity_m2": areas["finite_displacement_sensitivity_m2"],
        "calibration_displacement_m": areas["calibration_displacement_m"],
        "source_refinement_area_update": {
            "previous_area_m2": area_before,
            "refined_area_m2": areas["effective_area_m2"],
            "relative_change": (areas["effective_area_m2"] - area_before) / area_before,
            "basis": "recomputed from refined closed-lobe NHANAT triangles using the unchanged accepted respiratory basis",
        },
    })
    volumes = [
        abs(base_owner.signed_volume(rows[sid]["vertices6"][:, :3].astype(np.float64), rows[sid]["faces"]))
        for sid in (305, 306, 307, 308, 309)
    ]
    receipt["thorax_source_volume_m3"]["five_lung_envelopes"] = volumes
    receipt["thorax_source_volume_m3"]["sum"] = sum(volumes)
    for sid in sorted(split_edges):
        receipt["provenance"]["source_id_map"][str(sid)].setdefault("repair", {})["accepted_cycle_batch_refinement"] = {
            "method": "one-ring conforming edge-midpoint refinement; all cross-surface seed edges propagate through exact reciprocal lung/diaphragm faces",
            "source_face_indices": sorted(targets.get(sid, set())),
            "source_positions_modified": False,
            "source_volume_delta_m3": refinement[str(sid)]["source_volume_delta_m3"],
            "status": "candidate; full native accepted-frame regression pending",
        }
    receipt["qualification"]["diaphragm_lung_interface"] = (
        "candidate batch refinement for all 31 peak-phase outside-patch positive-segment pairs; "
        "narrow first witness reduced to point contact only; full accepted-frame patch and nonpatch regression pending"
    )
    receipt["qualification"]["inter_lobe_intersection"] = "not reassessed by the localized batch refinement"

    cardiac_identity_after = {sid: _surface_identity_sha256(sid, rows[sid]) for sid in cardiac_ids}
    if cardiac_identity_before != cardiac_identity_after:
        raise ValueError("batch refinement changed a cardiac source surface record")
    payload_bytes, vertex_count, index_count = _pack(header, rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    payload_path = output_dir / "resting-thorax.nhanatomy"
    payload_path.write_bytes(payload_bytes)
    payload_hash = sha(payload_path)
    cardiac_binding["output_anatomy_payload_sha256"] = payload_hash
    descendant = cardiac_binding.get("downstream_anatomy_refinement", {})
    descendant.update({
        "source_payload_sha256": BATCH_BASE_PAYLOAD_SHA256,
        "output_payload_sha256": payload_hash,
        "modified_stable_ids": sorted(set(descendant.get("modified_stable_ids", [])) | set(split_edges)),
        "preserved_cardiac_surface_sha256": {str(sid): cardiac_identity_after[sid] for sid in cardiac_ids},
        "preserved_cardiac_subset_byte_identity": True,
        "batch_refinement_evidence_sha256": BATCH_BASELINE_COMPARISON_SHA256,
    })
    cardiac_binding["downstream_anatomy_refinement"] = descendant
    receipt["payload"].update({
        "path": str(payload_path), "sha256": payload_hash, "surface_count": len(rows),
        "vertex_count": vertex_count, "index_count": index_count,
        "registration_fingerprint32": f"{header[5]:08x}",
    })
    receipt["functional_bindings"]["anatomy_payload_sha256"] = payload_hash
    receipt["provenance"]["conforming_cycle_batch_refinement"] = {
        "input_payload_path": str(BATCH_BASE_PAYLOAD),
        "input_payload_sha256": BATCH_BASE_PAYLOAD_SHA256,
        "input_receipt_path": str(BATCH_BASE_RECEIPT),
        "input_receipt_sha256": BATCH_BASE_RECEIPT_SHA256,
        "peak_witness_path": str(BATCH_PEAK_WITNESS),
        "peak_witness_sha256": BATCH_PEAK_WITNESS_SHA256,
        "baseline_comparison_path": str(BATCH_BASELINE_COMPARISON),
        "baseline_comparison_sha256": BATCH_BASELINE_COMPARISON_SHA256,
        "peak_step": 639,
        "baseline_step": 0,
        "phase_induced_segment_pair_count": len(selected),
        "target_face_pairs": [{"lung_stable_id": 305, "lung_face": a, "diaphragm_stable_id": 311, "diaphragm_face": b} for a, b in sorted(selected)],
        "source_face_to_candidate_face_indices": {sid: refinement[str(int(sid))]["source_face_to_candidate_face_indices"] for sid in ("305", "311")},
        "seed_face_counts": {str(sid): len(face_ids) for sid, face_ids in targets.items()},
        "split_edges_by_stable_id": {str(sid): [list(edge) for edge in sorted(edges)] for sid, edges in split_edges.items()},
        "surface_refinement": refinement,
        "reciprocal_patch_check": reciprocal_check,
        "source_vertex_positions_modified": False,
        "cardiac_surface_byte_identity": {str(sid): cardiac_identity_after[sid] for sid in cardiac_ids},
        "respiratory_effective_area_before_m2": area_before,
        "respiratory_effective_area_after_m2": areas["effective_area_m2"],
        "accepted_native_cycle_qualification": "pending",
    }
    receipt_path = output_dir / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    config = json.loads(BATCH_CONFIG.read_text())
    config["diaphragm_area_m2"] = areas["effective_area_m2"]
    config.setdefault("parameter_scope", {})["diaphragm_area_refinement_note"] = (
        "Updated from the exact refined closed-lobe geometry basis; inferred reference parameter, not measured subject data."
    )
    config_path = output_dir / "resting-reference-respiration.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    result = {
        "candidate_payload_path": str(payload_path),
        "candidate_payload_sha256": payload_hash,
        "candidate_receipt_path": str(receipt_path),
        "candidate_receipt_sha256": sha(receipt_path),
        "candidate_config_path": str(config_path),
        "candidate_config_sha256": sha(config_path),
        "vertex_count": vertex_count,
        "index_count": index_count,
        "selected_phase_induced_segment_pairs": len(selected),
        "seed_face_counts": {str(sid): len(face_ids) for sid, face_ids in targets.items()},
        "refinement": refinement,
        "respiratory_area_before_m2": area_before,
        "respiratory_area_after_m2": areas["effective_area_m2"],
        "reciprocal_patch_check": reciprocal_check,
    }
    (output_dir / "batch-refinement-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

if __name__ == "__main__": main()
