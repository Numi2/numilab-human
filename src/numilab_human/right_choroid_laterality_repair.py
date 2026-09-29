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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-payload', type=Path, required=True)
    parser.add_argument('--laterality-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
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
