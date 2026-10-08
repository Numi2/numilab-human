"""Excise one source-proved contralateral choroid component from a derived copy.

The original source 412 and source-support copy 587 remain byte-identical in
the packet. This appends a new inspection candidate with the eight isolated
right-choroid faces that occupy the left orbit explicitly excluded. It is a
lossy semantic source correction, not a generic mesh cleanup or clinical gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from . import anatomical_laterality as laterality
from . import model as human
from . import lung_envelope as lung
from . import surface_topology_audit as topology_audit
from . import whole_body_surface_gate as geometry_gate
from .cardiac_cavity_geometry import analyze_topology

SCHEMA = 'numi.human.right-choroid-contralateral-source-excision.v1'
PARENT_SHA = 'c8b8a01aac9d55d1dd154f0e3f92450e5a1a980668eebfcc2a012bb47268b280'
SOURCE_ID = 412
PARENT_COPY_ID = 587
NEW_ID = 599
EXCLUDED_PARENT_FACE_IDS = list(range(28852, 28860))
EXCLUDED_SOURCE_FACE_IDS = list(range(28862, 28870))
PAYLOAD_NAME = 'right-choroid-laterality-candidate.nhanatomy'
MANIFEST_NAME = 'right-choroid-laterality-candidate.manifest.json'


def require(ok, message):
    if not ok:
        raise human.ImportError('right choroid laterality repair: ' + message)


def derive(parent_payload, laterality_report):
    import numpy as np
    parent_payload = Path(parent_payload)
    require(human.sha256(parent_payload) == PARENT_SHA, 'published parent payload identity')
    _, header, records, vertices, indices = lung.decode(parent_payload)
    require(header[:2] == (5, 598), 'parent ABI/source count')
    evidence = next((r for r in laterality_report['rows'] if r['source_stable_id'] == SOURCE_ID), None)
    require(evidence is not None and evidence['visible_stable_id'] == PARENT_COPY_ID
            and evidence['source_member_id'] == 'FJ1337'
            and evidence['geometry_embedded_candidate'] is True
            and evidence['centroid_on_expected_side'] is True
            and evidence['wrong_side_selected_face_ids'] == EXCLUDED_PARENT_FACE_IDS
            and evidence['wrong_side_source_face_ids'] == EXCLUDED_SOURCE_FACE_IDS
            and evidence['wrong_side_vertex_count'] == 6,
            'selected contralateral source component changed')
    owner, fv, nv, fi, ni, stable, layer, reserved = map(int, records[PARENT_COPY_ID-1])
    require(stable == PARENT_COPY_ID and layer == 14 and reserved == 0 and ni == 28860*3,
            'parent choroid surface record')
    local = vertices[fv:fv+nv].copy()
    faces = (indices[fi:fi+ni].reshape(-1, 3)-fv).astype(np.int64)
    topology = analyze_topology(local[:, :3].astype(float).tolist(), faces.tolist())
    components = [set(group) for group in topology['face_components']]
    excluded = set(EXCLUDED_PARENT_FACE_IDS)
    require(topology['closed_oriented_manifold_candidate'] and len(components) == 2
            and excluded in components and len(next(group for group in components if group != excluded)) == 28852,
            'contralateral faces are not one isolated closed component')
    keep = np.array([i for i in range(len(faces)) if i not in excluded], dtype=np.int64)
    kept_faces = faces[keep]
    used = sorted(set(kept_faces.ravel().tolist()))
    remap = {old: new for new, old in enumerate(used)}
    new_faces = np.array([[remap[i] for i in f] for f in kept_faces], dtype='<u4')
    new_vertices = local[used].astype('<f4')
    require(len(new_vertices) == 14428 and len(new_faces) == 28852
            and np.array_equal(new_vertices, local[used]),
            'retained compiled source vertices moved')
    final_topology = analyze_topology(new_vertices[:, :3].astype(float).tolist(), new_faces.tolist())
    require(final_topology['closed_oriented_manifold_candidate']
            and len(final_topology['face_components']) == 1,
            'retained right choroid is not one closed component')
    exact = topology_audit.exact_embedding(new_vertices[:, :3].astype(float).tolist(),
                                           new_faces.tolist())
    require(exact['self_intersection_free'] and exact['count'] == 0,
            'retained choroid self-intersects')
    return {
        'header': header, 'records': records, 'vertices': vertices, 'indices': indices,
        'new_vertices': new_vertices, 'new_faces': new_faces, 'retained_parent_vertex_ids': used,
        'retained_parent_face_ids': keep.tolist(), 'topology': final_topology,
        'exact_embedding': exact, 'parent_owner': owner, 'parent_layer': layer,
        'source_face_ids': evidence['wrong_side_source_face_ids'],
        'area_fraction_removed': evidence['wrong_side_triangle_area_fraction'],
        'minimum_signed_midline_distance_m': evidence['minimum_signed_midline_distance_m'],
    }


def compose(parent_payload, laterality_report, output):
    import numpy as np
    d = derive(parent_payload, laterality_report)
    header = d['header']; records = d['records']; vertices = d['vertices']; indices = d['indices']
    nv, ni = len(vertices), len(indices)
    faces = d['new_faces']
    new_record = struct.pack('<8I', d['parent_owner'], nv, len(d['new_vertices']), ni,
                             faces.size, NEW_ID, d['parent_layer'], 0)
    raw = lung.HEADER.pack(b'NHANAT1\0', 5, NEW_ID,
        nv+len(d['new_vertices']), ni+faces.size, header[4], header[5])
    raw += records.tobytes()+new_record+vertices.tobytes()+d['new_vertices'].tobytes()
    raw += indices.tobytes()+(faces.ravel()+nv).astype('<u4').tobytes()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output/PAYLOAD_NAME
    path.write_bytes(raw)
    old_manifest_path = Path(parent_payload).with_name('source-topology-repair-candidates.manifest.json')
    old_manifest = human.read_json(old_manifest_path)
    ancestor = next(c for c in old_manifest['candidates'] if c['candidate_stable_id'] == PARENT_COPY_ID)
    source_face_ids = ancestor['source_derivation']['source_face_ids']
    require([source_face_ids[i] for i in EXCLUDED_PARENT_FACE_IDS] == EXCLUDED_SOURCE_FACE_IDS,
            'original source face ancestry changed')
    result = {
        'schema': SCHEMA, 'payload': {'sha256': human.sha256(path), 'abi': 5,
             'surfaces': NEW_ID, 'vertices': nv+len(d['new_vertices']), 'indices': ni+faces.size},
        'parent_payload_sha256': PARENT_SHA,
        'parent_candidate_manifest_sha256': human.sha256(old_manifest_path),
        'source_stable_id': SOURCE_ID, 'source_member_id': 'FJ1337',
        'parent_copy_stable_id': PARENT_COPY_ID, 'new_copy_stable_id': NEW_ID,
        'first_598_surface_records_vertices_indices_byte_identical': True,
        'excluded_parent_face_ids': EXCLUDED_PARENT_FACE_IDS,
        'excluded_original_source_face_ids': EXCLUDED_SOURCE_FACE_IDS,
        'retained_parent_vertex_ids': d['retained_parent_vertex_ids'],
        'retained_parent_face_ids': d['retained_parent_face_ids'],
        'excluded_component_edge_disconnected': True,
        'new_vertices_moved': False, 'new_faces_added': False, 'new_anatomical_members': 0,
        'removed_surface_area_fraction': d['area_fraction_removed'],
        'contralateral_minimum_signed_distance_m': d['minimum_signed_midline_distance_m'],
        'retained_topology': {k: len(v) if isinstance(v, list) else v
            for k, v in d['topology'].items() if k != 'face_components'},
        'retained_exact_embedding': {k: v for k, v in d['exact_embedding'].items()
            if k != 'triangle_pairs'},
        'clinical_anatomy': False, 'physical_volume': False, 'mechanics': False,
        'boundary': 'Exactly eight source-linked triangles in a separate contralateral right-choroid component are excluded from this new inspection copy. The original mesh and source-preserving copy remain in the payload. This is a semantic source-face excision, not proof of clinical eye anatomy, physical volume or mechanics.',
    }
    human.write_json(output/MANIFEST_NAME, result)
    return result


def audit_output(parent_payload, new_payload, manifest):
    import numpy as np
    require(human.sha256(parent_payload) == PARENT_SHA, 'parent hash drift')
    require(manifest['parent_candidate_manifest_sha256'] == human.sha256(
        Path(parent_payload).with_name('source-topology-repair-candidates.manifest.json')),
        'parent source-face ancestry manifest drift')
    _, ph, pr, pv, pi = lung.decode(parent_payload)
    _, h, r, v, i = lung.decode(new_payload)
    require(h[:2] == (5, NEW_ID) and h[4:] == ph[4:]
            and r[:598].tobytes() == pr.tobytes()
            and v[:len(pv)].tobytes() == pv.tobytes()
            and i[:len(pi)].tobytes() == pi.tobytes(),
            'prior 598 source and candidate meshes changed')
    original = pr[PARENT_COPY_ID-1]; added = r[NEW_ID-1]
    require(added.tolist() == [int(original[0]),len(pv),14428,len(pi),28852*3,
                               NEW_ID,int(original[6]),0], 'appended semantic record')
    used = manifest['retained_parent_vertex_ids']; retained = manifest['retained_parent_face_ids']
    source_vertices = pv[int(original[1]):int(original[1])+int(original[2])]
    source_faces = (pi[int(original[3]):int(original[3])+int(original[4])]
                    .reshape(-1, 3)-int(original[1]))
    remap = {old: new for new, old in enumerate(used)}
    expected_faces = np.array([[remap[x] for x in source_faces[j]] for j in retained], dtype=np.uint32)
    require(np.array_equal(v[len(pv):], source_vertices[used])
            and np.array_equal(i[len(pi):].reshape(-1, 3)-len(pv), expected_faces),
            'retained face/vertex ancestry differs')
    exact = topology_audit.exact_embedding(v[len(pv):, :3].astype(float).tolist(),
        (i[len(pi):].reshape(-1, 3)-len(pv)).tolist())
    require(exact['count'] == 0 and exact['topology']['closed_oriented_manifold_candidate']
            and exact['topology']['face_component_count'] == 1,
            'appended native candidate not closed and embedded')
    require(manifest['payload']['sha256'] == human.sha256(new_payload), 'manifest payload hash')
    return {'schema': SCHEMA+'.audit', 'passed': True, 'payload_sha256': human.sha256(new_payload),
            'first_598_byte_identical': True, 'source_parent': SOURCE_ID,
            'source_face_ids_excluded': EXCLUDED_SOURCE_FACE_IDS,
            'closed_single_component_exact_embedded': True,
            'clinical_anatomy': False, 'physical_volume': False, 'mechanics': False}


def audit_native_selected(base, indexed, quotient, parent_payload, source_proof,
                          selection, original_native_dir, original_pose_path,
                          new_payload, new_pack, new_pose_path):
    original = laterality.audit(base, indexed, quotient, parent_payload, source_proof,
                                selection, original_native_dir, original_pose_path)
    require(original['full_support_side_failure_source_ids'] == [412, 413, 426, 427]
            and original['geometry_embedded_but_laterality_failed_source_ids'] == [412],
            'baseline selected laterality finding changed')
    manifest = human.read_json(Path(new_payload).with_name(MANIFEST_NAME))
    source_audit = audit_output(parent_payload, new_payload, manifest)
    old_pose = human.read_json(original_pose_path)
    new_pose = human.read_json(new_pose_path)
    require(old_pose['bodies'] == new_pose['bodies'], 'native source-rest body pose changed')
    selection_doc = human.read_json(selection)
    hidden = sorted([*selection_doc['hidden_anatomy_stable_ids'], PARENT_COPY_ID])
    native_audit = topology_audit.audit_native(new_payload, new_pack, new_pose_path, 32767, hidden)
    gate = geometry_gate.audit(base, indexed, quotient, parent_payload,
                               source_proof, selection, original_native_dir)
    selected = [dict(row) for row in gate['rows']]
    require(selected[SOURCE_ID-1]['visible_stable_id'] == PARENT_COPY_ID
            and selected[SOURCE_ID-1]['selected_closed_embedded_surface_candidate'],
            'published source selection differs')
    selected[SOURCE_ID-1]['visible_stable_id'] = NEW_ID
    metadata, _ = laterality.source_metadata(Path(base).resolve())
    rows = laterality.inspect_selected(new_payload, selected, metadata, new_pose)
    require([row for row in rows if row['source_stable_id'] != SOURCE_ID] ==
            [row for row in original['rows'] if row['source_stable_id'] != SOURCE_ID],
            'unrelated bilateral surface result changed')
    fixed = next(row for row in rows if row['source_stable_id'] == SOURCE_ID)
    remaining = [row['source_stable_id'] for row in rows
                 if not row['all_selected_vertices_on_expected_side']]
    require(fixed['visible_stable_id'] == NEW_ID
            and fixed['all_selected_vertices_on_expected_side']
            and fixed['geometry_embedded_candidate']
            and remaining == [413, 426, 427],
            'new visible choroid did not close laterality defect')
    return {
        'schema': SCHEMA+'.native-selected-audit',
        'passed': True,
        'source_audit': source_audit,
        'native_audit': native_audit,
        'source_rest_body_pose_unchanged': True,
        'bilateral_source_surface_count': len(rows),
        'full_support_pass_before': original['full_support_side_pass_count'],
        'full_support_pass_after': len(rows)-len(remaining),
        'remaining_wrong_side_source_ids': remaining,
        'visible_source_412_stable_id': NEW_ID,
        'geometry_embedded_source_412': True,
        'all_599_surfaces_retained_in_native_packet': True,
        'clinical_anatomy': False, 'physical_volume': False, 'mechanics': False,
        'boundary': 'One selected eye reference is source-corrected for gross laterality in an actual native source-rest packet. Three right-eye references still have contralateral fragments; 521 noncohort source surfaces have no laterality verdict. No clinical eye anatomy or physical mechanics is admitted.',
    }



# Current integrated-payload mode: the earlier source proof fixes the 8 raw
# FJ1337 face IDs; this path binds them to 897 through its exact-cleanup receipt.
CURRENT_897_PARENT_SHA256 = "f38cdd8de7038bef2f4b30122aa2942f69c935afa472e948950948d2f2782633"
CURRENT_897_RECEIPT_SHA256 = "f574d0c4ef010552f774b7795ba8b958a3e6453ba8c33e420f02678518f19744"
CURRENT_897_CLEANUP_SHA256 = "becfaeefe1928bf8099492cc40e55660e15945e0b358b2649606d90d00afc2b0"
CURRENT_897_SOURCE_MEMBER_SHA256 = "db3077437828f311b6afa07dc9dc7a27ab0e005ddce62570420c61af31643f22"
CURRENT_897_PARTOF_ARCHIVE_SHA256 = "9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97"
CURRENT_897_SOURCE_FACE_IDS = list(range(28862, 28870))
CURRENT_897_PARENT_FACE_IDS = list(range(28852, 28860))
CURRENT_897_CLEANUP_REMOVED_FACE_IDS = [
    18447, 18448, 19953, 19965, 20401, 20402,
    23509, 23511, 24119, 24120, 28870, 28871,
]
CURRENT_897_OWNER_STABLE_ID = 382
CURRENT_897_CANDIDATE_STABLE_ID = 382
CURRENT_897_PAYLOAD_NAME = "resting-thorax.nhanatomy"
CURRENT_897_RECEIPT_NAME = "resting-anatomy-receipt.json"
CURRENT_897_MANIFEST_NAME = "right-choroid-current-897-candidate.manifest.json"
CURRENT_897_BASE_MANIFEST_NAME = "resting-anatomy-manifest.json"
CURRENT_897_AUDIT_NAME = "right-choroid-current-897-audit.json"


def _current_897_source_face_order(raw_faces, removed):
    removed = [int(x) for x in removed]
    require(len(removed) == len(set(removed))
            and all(0 <= x < int(raw_faces) for x in removed),
            "invalid exact-cleanup source face list")
    removed_set = set(removed)
    return [i for i in range(int(raw_faces)) if i not in removed_set]


def _current_897_compact_component(vertices6, faces, excluded):
    import numpy as np
    vertices6 = np.asarray(vertices6, dtype="<f4")
    faces = np.asarray(faces, dtype=np.int64)
    excluded = sorted(map(int, excluded))
    require(vertices6.ndim == 2 and vertices6.shape[1] == 6
            and np.isfinite(vertices6).all(), "stable-382 vertex field")
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces)
            and faces.min() >= 0 and faces.max() < len(vertices6),
            "stable-382 face field")
    require(excluded and len(excluded) == len(set(excluded))
            and excluded[0] >= 0 and excluded[-1] < len(faces),
            "excluded parent face IDs")
    excluded_vertices = set(faces[np.asarray(excluded)].reshape(-1).tolist())
    keep = np.asarray([i for i in range(len(faces)) if i not in set(excluded)],
                      dtype=np.int64)
    kept_faces = faces[keep]
    used = np.unique(kept_faces.reshape(-1))
    require(excluded_vertices.isdisjoint(set(used.tolist())),
            "excluded component shares parent vertex indices")
    require(kept_faces.min(initial=0) >= 0
            and (kept_faces.size == 0 or kept_faces.max() < len(vertices6)),
            "unbound candidate face vertex")
    # Keep the source Float32 vertex table and all surviving face-local indices
    # exactly as authored. Exact duplicate-position quotienting is only used by
    # topology/intersection audits; it never rewrites the emitted candidate.
    new_vertices = np.ascontiguousarray(vertices6, dtype="<f4")
    new_faces = np.ascontiguousarray(kept_faces, dtype=np.int64)
    require(new_vertices.tobytes() == vertices6.tobytes(),
            "retained Float32 positions/normals changed")
    return {
        "vertices6": new_vertices,
        "faces": new_faces,
        "retained_parent_vertex_ids": list(range(len(vertices6))),
        "retained_used_parent_vertex_ids": used.tolist(),
        "retained_parent_face_ids": keep.tolist(),
    }


def _current_897_coordinate_quotient(vertices6, faces):
    """Topology-only exact-coordinate weld of the retained Float32 position field."""
    from fractions import Fraction
    import numpy as np
    from .cardiac_cavity_geometry import analyze_topology, exact_coordinate_quotient
    positions = np.asarray(vertices6, dtype="<f4")[:, :3].astype(np.float64)
    keys = [tuple(Fraction.from_float(float(value)) for value in row)
            for row in positions]
    quotient = exact_coordinate_quotient({
        "vertices_mm": positions.tolist(),
        "coordinate_keys": keys,
        "triangles": np.asarray(faces, dtype=np.int64).tolist(),
    })
    source_to_quotient = quotient["source_vertex_to_vertex"]
    unique_positions = np.empty((quotient["topology"]["vertex_count"], 3), dtype=np.float64)
    assigned = np.zeros(len(unique_positions), dtype=bool)
    for source_index, quotient_index in enumerate(source_to_quotient):
        if not assigned[quotient_index]:
            unique_positions[quotient_index] = positions[source_index]
            assigned[quotient_index] = True
    require(bool(assigned.all()), "exact-coordinate quotient has an unassigned vertex")
    quotient_faces = np.asarray(quotient["triangles"], dtype=np.int64)
    used = np.unique(quotient_faces.reshape(-1))
    remap = np.full(len(unique_positions), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    compact_faces = remap[quotient_faces]
    compact_positions = unique_positions[used]
    topology = analyze_topology(compact_positions.tolist(), compact_faces.tolist())
    return {
        "method": "existing exact_coordinate_quotient owner over exact Fraction.from_float keys of serialized Float32 positions; zero geometric tolerance",
        "source_vertex_to_vertex": source_to_quotient,
        "identified_vertex_count": quotient["identified_vertex_count"],
        "unreferenced_quotient_vertex_count": int(len(unique_positions) - len(used)),
        "vertices": compact_positions,
        "faces": compact_faces,
        "topology": topology,
    }


def _current_897_derive(parent_payload, parent_receipt_path, sources):
    import hashlib
    import numpy as np
    from . import resting_anatomy_interface_patch as interface
    from .cardiac_cavity_geometry import analyze_topology

    parent_payload = Path(parent_payload).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    sources = Path(sources).resolve()
    require(parent_payload.is_file()
            and human.sha256(parent_payload) == CURRENT_897_PARENT_SHA256,
            "current 897 payload pin")
    require(parent_receipt_path.is_file()
            and human.sha256(parent_receipt_path) == CURRENT_897_RECEIPT_SHA256,
            "current 897 receipt pin")
    receipt = human.read_json(parent_receipt_path)
    meta = receipt.get("payload", {})
    require(meta.get("path") == str(parent_payload)
            and meta.get("sha256") == CURRENT_897_PARENT_SHA256
            and meta.get("abi") == 5 and meta.get("surface_count") == 524
            and meta.get("vertex_count") == 2088991
            and meta.get("index_count") == 11140047,
            "current 897 receipt payload declaration")
    header, rows = interface.parse_payload(parent_payload)
    require(header[:2] == (b"NHANAT1\0", 5)
            and header[2:5] == (524, 2088991, 11140047)
            and header[5] == int(meta["registration_fingerprint32"], 16)
            and header[6].hex() == meta["source_sha256"],
            "current 897 ABI, counts, registration, or source identity")

    source_map = receipt["provenance"]["source_id_map"][str(CURRENT_897_OWNER_STABLE_ID)]
    require(source_map.get("source_member") == "FJ1337"
            and source_map.get("source_sha256") == CURRENT_897_SOURCE_MEMBER_SHA256
            and source_map.get("body_index") == 23 and source_map.get("layer") == 1,
            "stable-382 FJ1337 owner identity")
    archives = receipt["provenance"]["bodyparts3d_archives"]
    archive_decl = next((x for x in archives if x.get("hierarchy") == "part_of"
                         and x.get("file") == "partof_BP3D_4.0_obj_99.zip"), None)
    require(archive_decl is not None
            and archive_decl.get("sha256") == CURRENT_897_PARTOF_ARCHIVE_SHA256,
            "receipt part_of archive declaration")
    archive_path, member, obj = human._bodyparts_obj_member(sources, "part_of", "FJ1337")
    require(human.sha256(archive_path) == CURRENT_897_PARTOF_ARCHIVE_SHA256
            and member == "partof_BP3D_4.0_obj_99/FJ1337.obj"
            and hashlib.sha256(obj).hexdigest() == CURRENT_897_SOURCE_MEMBER_SHA256,
            "raw FJ1337 part_of archive/member hashes")
    source_vertices, source_faces, _ = human._bodyparts_obj_geometry(obj, member)
    require(len(source_vertices) == 15685 and len(source_faces) == 28872,
            "raw FJ1337 vertex/face counts")

    cleanup = receipt["provenance"]["head_eye_source_exact_cleanup_candidate"]
    require(cleanup.get("schema") == "numi.human.head_eye_source_cleanup.v1"
            and cleanup.get("retained_static_audit_sha256") == CURRENT_897_CLEANUP_SHA256
            and CURRENT_897_OWNER_STABLE_ID in cleanup.get("changed_stable_ids", []),
            "current head/eye exact-cleanup receipt")
    cleanup_row = cleanup["rows"][str(CURRENT_897_OWNER_STABLE_ID)]
    require(cleanup_row.get("source_member") == "FJ1337"
            and cleanup_row.get("vertex_count") == 15685
            and cleanup_row.get("face_count_before") == 28872
            and cleanup_row.get("face_count_after") == 28860
            and cleanup_row.get("face_indices_removed") == CURRENT_897_CLEANUP_REMOVED_FACE_IDS
            and cleanup_row.get("exact_zero_area_face_indices") == []
            and cleanup_row.get("source_identity_preserved") is True
            and cleanup_row.get("vertex_position_normal_bytes_unchanged") is True,
            "current exact-cleanup face/vertex proof")
    parent_row = rows.get(CURRENT_897_OWNER_STABLE_ID)
    require(parent_row is not None and parent_row["body_index"] == 23
            and parent_row["layer"] == 1 and parent_row["flags"] == 0
            and len(parent_row["vertices6"]) == 15685
            and len(parent_row["faces"]) == 28860,
            "current stable-382 NHA surface")
    ancestry = _current_897_source_face_order(
        len(source_faces), cleanup_row["face_indices_removed"])
    require(len(ancestry) == len(parent_row["faces"])
            and np.array_equal(parent_row["faces"],
                               np.asarray(source_faces, dtype=np.int64)[ancestry]),
            "current stable-382 face order differs from raw source after recorded removals")
    excluded_parent = [i for i, source_id in enumerate(ancestry)
                       if source_id in set(CURRENT_897_SOURCE_FACE_IDS)]
    require(excluded_parent == CURRENT_897_PARENT_FACE_IDS,
            "source-proved face IDs do not map to current parent face IDs")

    parent_indexed_topology = analyze_topology(parent_row["vertices6"][:, :3].astype(float).tolist(),
                                               parent_row["faces"].tolist())
    parent_quotient = _current_897_coordinate_quotient(parent_row["vertices6"], parent_row["faces"])
    parent_topology = parent_quotient["topology"]
    components = [set(map(int, g)) for g in parent_topology["face_components"]]
    excluded_set = set(excluded_parent)
    require(parent_topology["closed_oriented_manifold_candidate"]
            and len(components) == 2 and excluded_set in components
            and len(next(g for g in components if g != excluded_set)) == 28852,
            "stable 382 is not the exact closed 8+28852 component pair")
    candidate = _current_897_compact_component(
        parent_row["vertices6"], parent_row["faces"], excluded_parent)
    require(len(candidate["vertices6"]) == len(parent_row["vertices6"])
            and len(candidate["faces"]) == 28852,
            "retained current right-choroid geometry counts")
    return {
        "parent_payload": parent_payload,
        "parent_receipt_path": parent_receipt_path,
        "parent_receipt": receipt,
        "header": header,
        "rows": rows,
        "record_order": _current_897_record_order(parent_payload, header[2]),
        "archive_path": archive_path,
        "member": member,
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "ancestry": ancestry,
        "cleanup_row": cleanup_row,
        "parent_topology": parent_topology,
        "parent_indexed_topology": parent_indexed_topology,
        "parent_coordinate_quotient": parent_quotient,
        "candidate": candidate,
    }


def _current_897_record_order(payload_path, count):
    from .resting_anatomy_interface_patch import HEADER, RECORD
    raw = Path(payload_path).read_bytes()
    if HEADER.unpack_from(raw)[2] != count:
        raise human.ImportError("right choroid 897: source record count changed")
    return [RECORD.unpack_from(raw, HEADER.size + i * RECORD.size)[5]
            for i in range(count)]


def _current_897_copy_cardiac_sidecars(parent_receipt_path, receipt, output):
    import shutil
    common = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    copied = {}
    for key in ("domain_boxes", "map", "polynomials"):
        entry = common[key]
        rel = Path(entry["path"])
        require(not rel.is_absolute() and ".." not in rel.parts and len(rel.parts) == 1,
                "common cardiac sidecar must remain a sibling")
        source = Path(parent_receipt_path).parent / rel
        require(source.is_file() and human.sha256(source) == entry["sha256"],
                "897 common cardiac sidecar hash " + key)
        target = Path(output) / rel
        shutil.copyfile(source, target)
        require(human.sha256(target) == entry["sha256"],
                "copied common cardiac sidecar hash " + key)
        copied[key] = {"path": rel.as_posix(), "sha256": entry["sha256"],
                       "bytes": target.stat().st_size}
    return copied


def _current_897_geometry_audit(parent_payload, parent_receipt_path, sources, candidate_payload, *, derived=None):
    import numpy as np
    from . import resting_anatomy_interface_patch as interface
    from . import resting_lung_edge_repair as lung_repair
    from .cardiac_cavity_geometry import analyze_topology

    d = derived if derived is not None else _current_897_derive(parent_payload, parent_receipt_path, sources)
    header, rows = interface.parse_payload(candidate_payload)
    require(header[:2] == (b"NHANAT1\0", 5) and header[2] == d["header"][2]
            and header[5:] == d["header"][5:],
            "candidate NHA header/source identity")
    order = _current_897_record_order(candidate_payload, header[2])
    require(order == d["record_order"] and len(set(order)) == 524
            and CURRENT_897_CANDIDATE_STABLE_ID in rows,
            "source surface order or corrected stable ID")
    for sid in d["record_order"]:
        before, after = d["rows"][sid], rows[sid]
        require((before["body_index"], before["layer"], before["flags"]) ==
                (after["body_index"], after["layer"], after["flags"])
                and before["vertices6"].tobytes() == after["vertices6"].tobytes(),
                "pre-existing anatomy vertex/owner record changed: " + str(sid))
        if sid != CURRENT_897_CANDIDATE_STABLE_ID:
            require(np.array_equal(before["faces"], after["faces"]),
                    "unrelated anatomy face geometry changed: " + str(sid))
    source = d["rows"][CURRENT_897_OWNER_STABLE_ID]
    actual = rows.get(CURRENT_897_CANDIDATE_STABLE_ID)
    expected = d["candidate"]
    require(actual is not None and actual["body_index"] == source["body_index"]
            and actual["layer"] == source["layer"] and actual["flags"] == source["flags"]
            and actual["vertices6"].tobytes() == source["vertices6"].tobytes()
            and np.array_equal(actual["faces"], expected["faces"]),
            "stable 382 is not the source-bound in-place 8-face correction")
    require(header[3] == d["header"][3]
            and header[4] == d["header"][4] - 8 * 3,
            "candidate vertex/index totals")
    indexed_topology = analyze_topology(actual["vertices6"][:, :3].astype(float).tolist(),
                                        actual["faces"].tolist())
    quotient = _current_897_coordinate_quotient(actual["vertices6"], actual["faces"])
    topology = quotient["topology"]
    exact = topology_audit.exact_embedding(
        quotient["vertices"].tolist(), quotient["faces"].tolist())
    require(topology["closed_oriented_manifold_candidate"]
            and len(topology["face_components"]) == 1
            and exact["self_intersection_free"] and exact["count"] == 0,
            "candidate must be exact-coordinate-welded closed, single-component, and self-intersection-free")
    from .resting_anatomy_interface_patch import HEADER, RECORD
    parent_raw, candidate_raw = Path(parent_payload).read_bytes(), Path(candidate_payload).read_bytes()
    parent_records = [RECORD.unpack_from(parent_raw, HEADER.size + i * RECORD.size)
                      for i in range(d["header"][2])]
    candidate_records = [RECORD.unpack_from(candidate_raw, HEADER.size + i * RECORD.size)
                         for i in range(header[2])]
    require(len(parent_records) == len(candidate_records) == 524,
            "surface record count changed")
    corrected_record_index = d["record_order"].index(CURRENT_897_OWNER_STABLE_ID)
    for record_index, (sid, before_record, after_record) in enumerate(
            zip(d["record_order"], parent_records, candidate_records, strict=True)):
        # Global first-index offsets after row 382 shift by 24, while local
        # geometry and every non-count owner field remain source-identical.
        before_record = tuple(map(int, before_record))
        after_record = tuple(map(int, after_record))
        require((before_record[0], before_record[1], before_record[2],
                 before_record[5], before_record[6], before_record[7]) ==
                (after_record[0], after_record[1], after_record[2],
                 after_record[5], after_record[6], after_record[7]),
                "surface record owner/vertex offsets/order changed: " + str(sid))
        expected_index_count = before_record[4] - (24 if sid == CURRENT_897_OWNER_STABLE_ID else 0)
        require(after_record[4] == expected_index_count,
                "surface record index count changed unexpectedly: " + str(sid))
        expected_start = before_record[3] - (24 if record_index > corrected_record_index else 0)
        require(after_record[3] == expected_start,
                "surface record index offset changed unexpectedly: " + str(sid))
    require(len(candidate_raw) ==
            HEADER.size + header[2] * RECORD.size + header[3] * 24 + header[4] * 4,
            "serialized corrected NHA byte length")
    area2 = np.linalg.norm(np.cross(
        source["vertices6"][source["faces"][:, 1], :3].astype(np.float64)
            - source["vertices6"][source["faces"][:, 0], :3].astype(np.float64),
        source["vertices6"][source["faces"][:, 2], :3].astype(np.float64)
            - source["vertices6"][source["faces"][:, 0], :3].astype(np.float64)), axis=1)
    removed_area_fraction = float(area2[CURRENT_897_PARENT_FACE_IDS].sum() / area2.sum())
    before_volume = lung_repair.signed_volume(
        source["vertices6"][:, :3].astype(np.float64), source["faces"])
    after_volume = lung_repair.signed_volume(
        actual["vertices6"][:, :3].astype(np.float64), actual["faces"])
    return {
        "schema": SCHEMA + ".current-897-geometry-audit",
        "passed": True,
        "parent_payload_sha256": human.sha256(d["parent_payload"]),
        "candidate_payload_sha256": human.sha256(candidate_payload),
        "parent_stable_id": CURRENT_897_OWNER_STABLE_ID,
        "corrected_stable_id": CURRENT_897_CANDIDATE_STABLE_ID,
        "source_member": "FJ1337",
        "source_member_sha256": CURRENT_897_SOURCE_MEMBER_SHA256,
        "excluded_source_face_ids": CURRENT_897_SOURCE_FACE_IDS,
        "excluded_parent_face_ids": CURRENT_897_PARENT_FACE_IDS,
        "candidate_vertices": len(actual["vertices6"]),
        "candidate_triangles": len(actual["faces"]),
        "surface_count_unchanged": True,
        "all_other_523_rows_local_geometry_and_all_vertex_tables_byte_identical": True,
        "stable_382_vertex_and_normal_table_byte_identical": True,
        "surface_record_offsets_rebuilt_by_existing_serializer": True,
        "raw_face_order_and_12_cleanup_deletions_verified": True,
        "indexed_parent_topology": {
            "method": "raw stable-382 vertex/face indices without welding",
            "component_count": d["parent_indexed_topology"]["face_component_count"],
            "closed_oriented_manifold_candidate": d["parent_indexed_topology"]["closed_oriented_manifold_candidate"],
        },
        "parent_component_face_counts_after_exact_coordinate_quotient": sorted(len(g) for g in
            d["parent_topology"]["face_components"]),
        "topology_weld": {
            "method": d["parent_coordinate_quotient"]["method"],
            "identified_duplicate_position_vertices": d["parent_coordinate_quotient"]["identified_vertex_count"],
            "tolerance": 0,
        },
        "corrected_stable_382_closed_single_component_after_exact_coordinate_quotient": True,
        "exact_self_intersection_free_after": True,
        "exact_self_intersection_pair_count": int(exact["count"]),
        "candidate_indexed_topology": {
            "method": "raw candidate vertex/face indices without welding",
            "component_count": indexed_topology["face_component_count"],
            "closed_oriented_manifold_candidate": indexed_topology["closed_oriented_manifold_candidate"],
        },
        "candidate_coordinate_quotient_topology": {
            k: (len(v) if isinstance(v, list) else v)
            for k, v in topology.items() if k != "face_components"
        },
        "candidate_coordinate_quotient_face_component_count": len(topology["face_components"]),
        "candidate_topology_weld_method": quotient["method"],
        "candidate_identified_duplicate_position_vertices": quotient["identified_vertex_count"],
        "removed_area_fraction_of_parent_stable_382": removed_area_fraction,
        "parent_signed_volume_m3_diagnostic_only": before_volume,
        "candidate_signed_volume_m3_diagnostic_only": after_volume,
        "signed_volume_delta_m3_diagnostic_only": after_volume - before_volume,
        "retained_positions_and_normals_byte_exact": True,
        "corrected_stable_382_remains_the_selected_source_identity": True,
        "clinical_anatomy": False, "physical_volume": False, "mechanics": False,
    }


def compose_current_897(parent_payload, parent_receipt_path, sources, output):
    d = _current_897_derive(parent_payload, parent_receipt_path, sources)
    return _compose_current_897_derived(d, sources, output)


def _verify_disjoint_lung_rows(reference_rows, parent_rows):
    """Require a lung-only change before retaining the source-proved eye repair."""
    import numpy as np
    require(set(reference_rows) == set(parent_rows), "lung composition changed anatomical identity set")
    changed = []
    for sid, before in reference_rows.items():
        after = parent_rows[sid]
        require(all(before[key] == after[key] for key in ("body_index", "layer", "flags")),
                "lung composition changed owner fields: " + str(sid))
        same = (before["vertices6"].dtype == after["vertices6"].dtype
                and before["vertices6"].shape == after["vertices6"].shape
                and before["vertices6"].tobytes() == after["vertices6"].tobytes()
                and np.array_equal(before["faces"], after["faces"]))
        if not same:
            require(sid in range(305, 312), "lung composition changed a non-lung row: " + str(sid))
            changed.append(int(sid))
    return sorted(changed)


def compose_after_lung_correction(parent_payload, parent_receipt_path, sources, output, *,
                                  reference_payload, reference_receipt,
                                  expected_parent_sha256, expected_parent_receipt_sha256):
    """Apply the retained 897 choroid correction after disjoint lung preparation.

    The original source/face/laterality proof remains mandatory. Only rows
    305--311 may differ from its pinned parent; the current lung receipt and
    all of its functional geometry metadata remain the composition authority.
    This checks composition, not the lung candidate's physical admission.
    """
    import copy
    from . import resting_anatomy_interface_patch as interface
    parent_payload = Path(parent_payload).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    require(human.sha256(parent_payload) == expected_parent_sha256,
            "lung composition parent payload hash")
    require(human.sha256(parent_receipt_path) == expected_parent_receipt_sha256,
            "lung composition parent receipt hash")
    d = _current_897_derive(reference_payload, reference_receipt, sources)
    header, rows = interface.parse_payload(parent_payload)
    require(header[:3] == d["header"][:3] and header[5:] == d["header"][5:],
            "lung composition changed NHA ABI, registration, or source identity")
    order = _current_897_record_order(parent_payload, header[2])
    require(order == d["record_order"], "lung composition changed source record order")
    changed = _verify_disjoint_lung_rows(d["rows"], rows)
    receipt = human.read_json(parent_receipt_path)
    meta = receipt["payload"]
    require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1"
            and meta.get("path") == str(parent_payload)
            and meta.get("sha256") == expected_parent_sha256
            and meta.get("abi") == 5 and meta.get("surface_count") == header[2]
            and meta.get("vertex_count") == header[3] and meta.get("index_count") == header[4]
            and int(meta.get("registration_fingerprint32", "-1"), 16) == header[5]
            and meta.get("source_sha256") == header[6].hex()
            and receipt["functional_bindings"].get("anatomy_payload_sha256") == expected_parent_sha256,
            "lung composition receipt does not bind its actual parent payload")
    ref_prov = d["parent_receipt"]["provenance"]
    prov = receipt["provenance"]
    require(prov["head_eye_source_exact_cleanup_candidate"] == ref_prov["head_eye_source_exact_cleanup_candidate"],
            "lung composition changed the source eye cleanup proof")
    require(set(prov["source_id_map"]) == set(ref_prov["source_id_map"]),
            "lung composition changed source map identities")
    for sid, row in ref_prov["source_id_map"].items():
        if int(sid) not in range(305, 312):
            require(prov["source_id_map"][sid] == row,
                    "lung composition changed non-lung source provenance: " + sid)
    cardiac = copy.deepcopy(prov["cardiac_geometry_binding"])
    reference_cardiac = copy.deepcopy(ref_prov["cardiac_geometry_binding"])
    require(cardiac["common_field"]["anatomy_payload_sha256"] == expected_parent_sha256,
            "lung composition cardiac binding does not bind its parent")
    cardiac["common_field"]["anatomy_payload_sha256"] = CURRENT_897_PARENT_SHA256
    require(cardiac == reference_cardiac, "lung composition changed unrelated cardiac binding")
    d.update(parent_payload=parent_payload, parent_receipt_path=parent_receipt_path,
             parent_receipt=receipt, header=header, rows=rows, record_order=order)
    d["disjoint_lung_composition"] = {
        "reference_payload": {"path": str(Path(reference_payload).resolve()), "sha256": CURRENT_897_PARENT_SHA256},
        "reference_receipt": {"path": str(Path(reference_receipt).resolve()), "sha256": CURRENT_897_RECEIPT_SHA256},
        "changed_parent_rows": changed,
        "all_nonlung_rows_and_source_provenance_match_source_proved_reference": True,
        "retained_lung_receipt_is_functional_geometry_authority": True,
        "lung_candidate_admission": "not implied by source-disjoint composition",
    }
    result = _compose_current_897_derived(d, sources, output)
    require(human.sha256(parent_payload) == expected_parent_sha256
            and human.sha256(parent_receipt_path) == expected_parent_receipt_sha256,
            "lung composition inputs changed during preparation")
    return result


def _compose_current_897_derived(d, sources, output):
    import copy
    import hashlib
    from . import resting_lung_edge_repair as lung_repair

    output = Path(output).resolve()
    require(not output.exists(), "output directory already exists")
    parent_payload_sha = human.sha256(d["parent_payload"])
    parent_receipt_sha = human.sha256(d["parent_receipt_path"])
    base_manifest_path = d["parent_receipt_path"].with_name(CURRENT_897_BASE_MANIFEST_NAME)
    require(base_manifest_path.is_file(), "current 897 valid anatomy manifest")
    base_manifest_raw = base_manifest_path.read_bytes()
    base_manifest = human.read_json(base_manifest_path)
    require(base_manifest.get("schema") == "numi.human.resting-anatomy-manifest.v1"
            and base_manifest.get("receipt", {}).get("path") == str(d["parent_receipt_path"])
            and base_manifest.get("receipt", {}).get("sha256") == parent_receipt_sha
            and base_manifest.get("payload") == d["parent_receipt"].get("payload")
            and base_manifest.get("functional_bindings") == d["parent_receipt"].get("functional_bindings"),
            "current 897 anatomy manifest does not bind receipt/payload/function identities")
    require(CURRENT_897_CANDIDATE_STABLE_ID == CURRENT_897_OWNER_STABLE_ID
            and len(d["record_order"]) == 524,
            "in-place correction identity or parent record order")
    rows = {
        sid: {"body_index": row["body_index"], "layer": row["layer"],
              "flags": row["flags"], "vertices6": row["vertices6"].copy(),
              "faces": row["faces"].copy()}
        for sid, row in d["rows"].items()
    }
    owner = rows[CURRENT_897_OWNER_STABLE_ID]
    owner["faces"] = d["candidate"]["faces"]
    output.mkdir(parents=True, exist_ok=False)
    payload_path = output / CURRENT_897_PAYLOAD_NAME
    record_order = list(d["record_order"])
    payload_path.write_bytes(lung_repair._serialize_payload(
        d["header"], record_order, rows))
    sidecars = _current_897_copy_cardiac_sidecars(
        d["parent_receipt_path"], d["parent_receipt"], output)
    geometry_audit = _current_897_geometry_audit(
        d["parent_payload"], d["parent_receipt_path"], sources, payload_path, derived=d)
    manifest = {
        "schema": SCHEMA + ".current-897-manifest",
        "parent_payload": {"path": str(d["parent_payload"]),
                           "sha256": parent_payload_sha},
        "parent_receipt": {"path": str(d["parent_receipt_path"]),
                           "sha256": parent_receipt_sha},
        "exact_cleanup": {
            "schema": d["parent_receipt"]["provenance"]["head_eye_source_exact_cleanup_candidate"]["schema"],
            "retained_static_audit_sha256": CURRENT_897_CLEANUP_SHA256,
            "row_stable_id": CURRENT_897_OWNER_STABLE_ID,
            "removed_source_face_ids": CURRENT_897_CLEANUP_REMOVED_FACE_IDS,
            "raw_face_order_matches_parent_after_removals": True,
        },
        "source": {
            "member_id": "FJ1337", "archive_hierarchy": "part_of",
            "archive_path": str(d["archive_path"]),
            "archive_sha256": CURRENT_897_PARTOF_ARCHIVE_SHA256,
            "member_name": d["member"],
            "member_sha256": CURRENT_897_SOURCE_MEMBER_SHA256,
            "raw_vertex_count": len(d["source_vertices"]),
            "raw_face_count": len(d["source_faces"]),
        },
        "excision": {
            "parent_stable_id_preserved": CURRENT_897_OWNER_STABLE_ID,
            "corrected_stable_id": CURRENT_897_CANDIDATE_STABLE_ID,
            "excluded_parent_face_ids": CURRENT_897_PARENT_FACE_IDS,
            "excluded_raw_source_face_ids": CURRENT_897_SOURCE_FACE_IDS,
            "operation": "replace stable 382 local face list with the same ordered source faces except the eight source-proved contralateral faces; retain its full vertex/normal table and all 524 stable IDs",
            "vertices_moved": False, "normals_changed": False,
            "new_faces_added": False, "new_anatomical_members": 0,
            "corrected_stable_id_remains_source_and_function_selected": True,
        },
        "geometry_audit": geometry_audit,
        "relative_cardiac_sidecars": sidecars,
        "output_payload": {
            "path": str(payload_path), "sha256": human.sha256(payload_path),
            "abi": 5, "surface_count": int(d["header"][2]),
            "vertex_count": int(d["header"][3]),
            "index_count": int(d["header"][4] - 8 * 3),
        },
        "limitations": {
            "source_laterality_basis": "The existing laterality owner’s source-proved eight raw FJ1337 face IDs are bound to current row 382 by raw OBJ face order and the current receipt’s exact 12-face cleanup list; no proximity heuristic is used.",
            "clinical_anatomy": False, "physical_volume": False, "mechanics": False,
        },
    }
    if "disjoint_lung_composition" in d:
        manifest["disjoint_lung_composition"] = d["disjoint_lung_composition"]
    manifest_path = output / CURRENT_897_MANIFEST_NAME
    human.write_json(manifest_path, manifest)
    manifest_sha = human.sha256(manifest_path)
    parent_receipt = d["parent_receipt"]
    updated = copy.deepcopy(parent_receipt)
    previous_input_sha = updated["payload"].get("input_payload_sha256")
    out_header = (b"NHANAT1\0", 5, int(d["header"][2]), manifest["output_payload"]["vertex_count"],
                  manifest["output_payload"]["index_count"], d["header"][5], d["header"][6])
    output_sha = human.sha256(payload_path)
    updated["payload"].update({
        "path": str(payload_path), "input_payload_sha256": parent_payload_sha,
        "sha256": output_sha, "surface_count": out_header[2],
        "vertex_count": out_header[3], "index_count": out_header[4],
    })
    require(updated["functional_bindings"].get("anatomy_payload_sha256")
            == parent_payload_sha,
            "parent functional anatomy payload binding")
    updated["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    common_field = updated.get("provenance", {}).get("cardiac_geometry_binding", {}).get("common_field")
    require(isinstance(common_field, dict)
            and common_field.get("anatomy_payload_sha256") == parent_payload_sha,
            "common cardiac field does not bind selected parent NHA")
    common_field["anatomy_payload_sha256"] = output_sha
    source_map_row = updated["provenance"]["source_id_map"][str(CURRENT_897_OWNER_STABLE_ID)]
    repair_provenance = source_map_row.get("repair")
    if repair_provenance is None:
        repair_provenance = {}
        source_map_row["repair"] = repair_provenance
    require(isinstance(repair_provenance, dict),
            "stable-382 source-map repair field is not composable")
    repair_provenance["right_choroid_source_face_excision"] = {
        "schema": SCHEMA,
        "source_face_ids_excluded": CURRENT_897_SOURCE_FACE_IDS,
        "parent_face_ids_excluded": CURRENT_897_PARENT_FACE_IDS,
        "source_archive_sha256": CURRENT_897_PARTOF_ARCHIVE_SHA256,
        "source_member_sha256": CURRENT_897_SOURCE_MEMBER_SHA256,
        "parent_payload_sha256": parent_payload_sha,
        "derived_payload_sha256": output_sha,
        "status": "eight_source_proved_contralateral_faces_removed_from_same_selected_stable_id",
    }
    updated["provenance"]["right_choroid_laterality_candidate_897"] = {
        "schema": SCHEMA,
        "status": "derived_in_place_source_corrected_row_exact_closed_single_component",
        "parent_payload_path": str(d["parent_payload"]),
        "parent_payload_sha256": parent_payload_sha,
        "parent_receipt_path": str(d["parent_receipt_path"]),
        "parent_receipt_sha256": parent_receipt_sha,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": manifest_sha,
        "source_stable_id_preserved": CURRENT_897_OWNER_STABLE_ID,
        "corrected_stable_id": CURRENT_897_CANDIDATE_STABLE_ID,
        "source_member": "FJ1337",
        "source_archive_hierarchy": "part_of",
        "source_archive_sha256": CURRENT_897_PARTOF_ARCHIVE_SHA256,
        "source_member_sha256": CURRENT_897_SOURCE_MEMBER_SHA256,
        "excluded_source_face_ids": CURRENT_897_SOURCE_FACE_IDS,
        "excluded_parent_face_ids": CURRENT_897_PARENT_FACE_IDS,
        "inherited_payload_input_sha256": previous_input_sha,
        "corrected_stable_382_remains_the_selected_source_identity": True,
        "new_anatomical_member": False, "clinical_anatomy": False,
        "physical_volume": False, "mechanics": False,
        "boundary": "Stable 382 remains the selected FJ1337 identity. This derived payload replaces only that row's face list by removing the eight source-proved contralateral FJ1337 faces; its full Float32 vertex/normal table and all other rows remain unchanged. The unmodified 897 payload is retained as source input."
    }
    receipt_path = output / CURRENT_897_RECEIPT_NAME
    human.write_json(receipt_path, updated)

    standard_manifest = copy.deepcopy(base_manifest)
    standard_manifest["payload"] = copy.deepcopy(updated["payload"])
    standard_manifest["functional_bindings"] = copy.deepcopy(updated["functional_bindings"])
    standard_manifest["receipt"] = {
        "path": str(receipt_path),
        "sha256": human.sha256(receipt_path),
    }
    standard_manifest["mass_geometry_accounting"] = copy.deepcopy(updated["mass_geometry_accounting"])
    standard_manifest["source_surfaces"] = copy.deepcopy(updated["provenance"]["source_id_map"])
    standard_manifest["source_receipt_lineage"] = {
        "composition": "right_choroid_laterality_candidate_897",
        "source_receipt_path": str(d["parent_receipt_path"]),
        "source_receipt_sha256": parent_receipt_sha,
        "source_manifest_path": str(base_manifest_path),
        "source_manifest_sha256": hashlib.sha256(base_manifest_raw).hexdigest(),
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": manifest_sha,
        "derived_candidate_is_not_a_new_source_surface": True,
    }
    standard_manifest_path = output / CURRENT_897_BASE_MANIFEST_NAME
    human.write_json(standard_manifest_path, standard_manifest)
    require(standard_manifest["schema"] == "numi.human.resting-anatomy-manifest.v1"
            and standard_manifest["receipt"] == {
                "path": str(receipt_path), "sha256": human.sha256(receipt_path)}
            and standard_manifest["payload"] == updated["payload"]
            and standard_manifest["functional_bindings"] == updated["functional_bindings"],
            "emitted resting-anatomy-manifest.v1 does not bind output receipt")
    for key in ("domain_boxes", "map", "polynomials"):
        sidecar = updated["provenance"]["cardiac_geometry_binding"]["common_field"][key]
        sidecar_path = output / sidecar["path"]
        require(sidecar_path.is_file() and human.sha256(sidecar_path) == sidecar["sha256"],
                "output receipt cardiac sidecar is unavailable: " + key)

    expected_receipt = copy.deepcopy(parent_receipt)
    expected_receipt["payload"].update(updated["payload"])
    expected_receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    expected_receipt["provenance"]["cardiac_geometry_binding"]["common_field"]["anatomy_payload_sha256"] = output_sha
    expected_receipt["provenance"]["source_id_map"][str(CURRENT_897_OWNER_STABLE_ID)] = \
        copy.deepcopy(updated["provenance"]["source_id_map"][str(CURRENT_897_OWNER_STABLE_ID)])
    expected_receipt["provenance"]["right_choroid_laterality_candidate_897"] = \
        updated["provenance"]["right_choroid_laterality_candidate_897"]
    actual_receipt = human.read_json(receipt_path)
    require(actual_receipt == expected_receipt,
            "updated receipt changed undeclared source or functional fields")
    require(actual_receipt["mass_geometry_accounting"] == parent_receipt["mass_geometry_accounting"]
            and actual_receipt["thorax_source_volume_m3"] == parent_receipt["thorax_source_volume_m3"]
            and actual_receipt["derived_lobe_overlap_partition"] == parent_receipt["derived_lobe_overlap_partition"]
            and all(actual_receipt["provenance"]["source_id_map"][sid] ==
                    parent_receipt["provenance"]["source_id_map"][sid]
                    for sid in parent_receipt["provenance"]["source_id_map"]
                    if sid != str(CURRENT_897_OWNER_STABLE_ID))
            and actual_receipt["provenance"]["source_id_map"][str(CURRENT_897_OWNER_STABLE_ID)]["source_sha256"] ==
                parent_receipt["provenance"]["source_id_map"][str(CURRENT_897_OWNER_STABLE_ID)]["source_sha256"]
            and actual_receipt["provenance"]["head_eye_source_exact_cleanup_candidate"] ==
                parent_receipt["provenance"]["head_eye_source_exact_cleanup_candidate"],
            "candidate changed lung, mass, source identity, or historical cleanup data")
    audit = {
        "schema": SCHEMA + ".current-897-final-audit",
        "passed": geometry_audit["passed"],
        "geometry": geometry_audit,
        "receipt": {
            "passed": True,
            "sha256": human.sha256(receipt_path),
            "parent_sha256": parent_receipt_sha,
            "functional_anatomy_sha_updated": True,
            "surface_count_unchanged": True,
            "corrected_stable_id": CURRENT_897_OWNER_STABLE_ID,
            "lung_mass_and_source_metadata_unchanged": True,
            "relative_cardiac_sidecars_hash_verified": True,
        },
        "candidate_manifest_sha256": manifest_sha,
        "standard_manifest_path": str(standard_manifest_path),
        "standard_manifest_sha256": human.sha256(standard_manifest_path),
        "standard_manifest_schema": standard_manifest["schema"],
        "payload_sha256": output_sha,
        "sidecars": sidecars,
    }
    human.write_json(output / CURRENT_897_AUDIT_NAME, audit)
    return manifest



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-payload', type=Path, required=True)
    parser.add_argument('--laterality-report', type=Path)
    parser.add_argument('--parent-receipt', type=Path)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.parent_receipt is not None or args.sources is not None:
        require(args.parent_receipt is not None and args.sources is not None
                and args.laterality_report is None,
                "current-897 mode requires --parent-receipt and --sources only")
        result = compose_current_897(args.parent_payload, args.parent_receipt,
                                     args.sources, args.output)
        print(json.dumps({
            "corrected_stable_id": CURRENT_897_CANDIDATE_STABLE_ID,
            "excluded_parent_face_ids": CURRENT_897_PARENT_FACE_IDS,
            "excluded_source_face_ids": CURRENT_897_SOURCE_FACE_IDS,
            "closed_single_component_exact_embedded": True,
            "payload_sha256": result["output_payload"]["sha256"],
            "receipt_path": str(args.output.resolve() / CURRENT_897_RECEIPT_NAME),
        }), flush=True)
        return
    require(args.laterality_report is not None,
            "legacy mode requires --laterality-report")
    evidence = human.read_json(args.laterality_report)
    require(evidence['schema'] == laterality.SCHEMA
            and evidence['selected_payload_sha256'] == PARENT_SHA,
            'selected laterality evidence identity')
    result = compose(args.parent_payload, evidence, args.output)
    checked = audit_output(args.parent_payload, args.output/PAYLOAD_NAME, result)
    human.write_json(args.output/'right-choroid-laterality-audit.json', checked)
    print(json.dumps({'candidate_id': NEW_ID,
        'removed_source_face_ids': EXCLUDED_SOURCE_FACE_IDS,
        'closed_single_component_exact_embedded': True,
        'payload_sha256': result['payload']['sha256']}), flush=True)


if __name__ == '__main__':
    main()
