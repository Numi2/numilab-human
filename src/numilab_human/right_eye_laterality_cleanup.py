"""Excise three isolated contralateral right-eye source components.

Source-rest placement is corrected in derived inspection copies only. The raw
cornea, sclera and lens-ligament meshes remain byte-identical in the packet.
Their unresolved topology/self-intersection defects remain explicit failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

from . import anatomical_laterality as laterality
from . import lung_envelope as lung
from . import model as human
from . import right_choroid_laterality_repair as choroid
from . import surface_topology_audit as topology_audit
from . import whole_body_surface_gate as geometry_gate
from .cardiac_cavity_geometry import analyze_topology
from .compiled_quotient_embeddedness import classify_quotient

SCHEMA = 'numi.human.right-eye-contralateral-source-components.v1'
PARENT_599_SHA = 'c32c13c555308f9671c92f1083db4f9fa4a4b096cd3d1734412013c0c7a47ecd'
PAYLOAD_NAME = 'right-eye-laterality-candidates.nhanatomy'
MANIFEST_NAME = 'right-eye-laterality-candidates.manifest.json'
SPECS = (
    (413, 600, 'FJ1340', tuple(range(4588, 4606))),
    (426, 601, 'FJ1368', tuple(range(39520, 39524))),
    (427, 602, 'FJ1371', tuple(range(7372, 7376))),
)


def require(ok, message):
    if not ok:
        raise human.ImportError('right eye laterality cleanup: ' + message)


def derive(parent_payload, pose_path):
    import numpy as np
    require(human.sha256(parent_payload) == PARENT_599_SHA, '599-surface parent payload identity')
    _, header, records, vertices, indices = lung.decode(parent_payload)
    require(header[:2] == (5, 599), '599-surface parent ABI')
    pose = human.read_json(pose_path)
    require(pose['surface_count'] == 599 and pose['schema'] ==
            'numi.human.native-torso-anatomy-pose-snapshot.v1', '599-surface native pose identity')
    bodies = {x['body_index']: x for x in pose['bodies']}
    require(len(bodies) == len(pose['bodies']) and 20 in bodies, 'source-rest owner coverage')
    torso = bodies[20]
    torso_origin = np.asarray(torso['position_world_m'], float)
    left_axis = laterality.quaternion_rotation(torso['orientation_world_xyzw']) @ np.array([0., 0., -1.])
    require(float(left_axis[0]) > .999, 'source-rest sagittal axis changed')
    results = []
    for source_id, new_id, member, excluded_tuple in SPECS:
        owner, fv, nv, fi, ni, stable, layer, reserved = map(int, records[source_id-1])
        require(stable == source_id and layer == 14 and reserved == 0
                and owner in bodies and ni % 3 == 0, 'right-eye source owner/type')
        local = vertices[fv:fv+nv].copy()
        faces = (indices[fi:fi+ni].reshape(-1, 3)-fv).astype(np.int64)
        body = bodies[owner]
        world = local[:, :3].astype(float) @ laterality.quaternion_rotation(
            body['orientation_world_xyzw']).T + body['position_world_m']
        original = laterality.support_metrics(world, faces, torso_origin, left_axis,
                                               -1, list(range(len(faces))))
        excluded = set(excluded_tuple)
        require(original['centroid_on_expected_side']
                and original['wrong_side_selected_face_ids'] == list(excluded_tuple)
                and original['wrong_side_source_face_ids'] == list(excluded_tuple)
                and original['wrong_side_triangle_area_fraction'] < 1e-5,
                'source contralateral component differs')
        source_topology = analyze_topology(local[:, :3].astype(float).tolist(), faces.tolist())
        components = [set(group) for group in source_topology['face_components']]
        require(sum(len(group) for group in components) == len(faces)
                and all(group <= excluded or group.isdisjoint(excluded) for group in components),
                'wrong-side source faces share a connected component with eye support')
        used_wrong = set(faces[list(excluded)].ravel().tolist())
        retained_face_ids = [j for j in range(len(faces)) if j not in excluded]
        retained = faces[retained_face_ids]
        used_right = set(retained.ravel().tolist())
        require(used_wrong.isdisjoint(used_right), 'contralateral component shares source vertex IDs')
        require({tuple(local[j, :3]) for j in used_wrong}.isdisjoint(
                {tuple(local[j, :3]) for j in used_right}),
                'contralateral component shares exact spatial source vertices')
        used = sorted(used_right)
        remap = {old: new for new, old in enumerate(used)}
        candidate_faces = np.asarray([[remap[j] for j in f] for f in retained], dtype='<u4')
        candidate_vertices = local[used].astype('<f4')
        placed = laterality.support_metrics(world[used], candidate_faces,
            torso_origin, left_axis, -1, retained_face_ids)
        require(placed['all_selected_vertices_on_expected_side']
                and placed['wrong_side_triangle_count'] == 0,
                'derived right-eye copy still crosses the midline')
        quotient = classify_quotient(candidate_vertices[:, :3].astype(float).tolist(),
                                      candidate_faces.tolist())
        results.append({
            'source_stable_id': source_id, 'candidate_stable_id': new_id,
            'source_member_id': member, 'core_body_index': owner, 'layer_code': layer,
            'excluded_original_source_face_ids': list(excluded_tuple),
            'excluded_edge_component_count': sum(group <= excluded for group in components),
            'removed_surface_area_fraction': original['wrong_side_triangle_area_fraction'],
            'source_vertex_count': len(local), 'source_triangle_count': len(faces),
            'candidate_vertex_count': len(candidate_vertices),
            'candidate_triangle_count': len(candidate_faces),
            'retained_vertex_ids_sha256': hashlib.sha256(
                np.asarray(used, dtype='<u4').tobytes()).hexdigest(),
            'retained_face_ids_sha256': hashlib.sha256(
                np.asarray(retained_face_ids, dtype='<u4').tobytes()).hexdigest(),
            'compiled_quotient_status_after': quotient['status'],
            'compiled_quotient_exact_intersection_pairs_after': quotient['exact_intersection_pairs'],
            'compiled_quotient_closed_after': quotient['topology']['closed_oriented_manifold_candidate'],
            'source_geometry_qualified': False,
            'local_vertices': candidate_vertices, 'local_faces': candidate_faces,
        })
    return header, records, vertices, indices, results


def compose(parent_payload, pose_path, output):
    import numpy as np
    header, records, vertices, indices, rows = derive(parent_payload, pose_path)
    nv, ni = len(vertices), len(indices)
    new_records, vertex_parts, index_parts = [], [], []
    for row in rows:
        v, f = row.pop('local_vertices'), row.pop('local_faces')
        new_records.append(struct.pack('<8I', row['core_body_index'], nv, len(v), ni,
                                       f.size, row['candidate_stable_id'], row['layer_code'], 0))
        vertex_parts.append(v.tobytes())
        index_parts.append((f.ravel()+nv).astype('<u4').tobytes())
        nv += len(v); ni += f.size
    raw = lung.HEADER.pack(b'NHANAT1\0', 5, 602, nv, ni, header[4], header[5])
    raw += records.tobytes()+b''.join(new_records)+vertices.tobytes()+b''.join(vertex_parts)
    raw += indices.tobytes()+b''.join(index_parts)
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    path = output/PAYLOAD_NAME;path.write_bytes(raw)
    manifest = {
        'schema': SCHEMA, 'payload': {'sha256': human.sha256(path), 'abi': 5,
            'surfaces': 602, 'vertices': nv, 'indices': ni},
        'parent_599_payload_sha256': PARENT_599_SHA,
        'parent_599_manifest_sha256': human.sha256(Path(parent_payload).with_name(choroid.MANIFEST_NAME)),
        'source_rest_pose_sha256': human.sha256(pose_path),
        'first_599_records_vertices_indices_byte_identical': True,
        'new_anatomical_members': 0,
        'candidates': rows,
        'source_face_excision': True, 'new_vertices_moved': False, 'new_faces_added': False,
        'clinical_anatomy': False, 'physical_volume': False, 'mechanics': False,
        'boundary': 'Three complete contralateral source face components are excluded from separate inspection copies. All 599 prior surfaces are retained byte-identically. Each retained source mesh still has unresolved topology or intersection failures; no clinical or physical anatomy is admitted.',
    }
    human.write_json(output/MANIFEST_NAME, manifest)
    return manifest


def audit_output(parent_payload, pose_path, payload, manifest):
    import numpy as np
    require(human.sha256(parent_payload) == manifest['parent_599_payload_sha256'] == PARENT_599_SHA
            and human.sha256(pose_path) == manifest['source_rest_pose_sha256']
            and human.sha256(Path(parent_payload).with_name(choroid.MANIFEST_NAME)) ==
                manifest['parent_599_manifest_sha256'], 'parent source identity')
    _, ph, pr, pv, pi = lung.decode(parent_payload)
    _, h, records, vertices, indices = lung.decode(payload)
    require(h[:2] == (5, 602) and h[4:] == ph[4:]
            and records[:599].tobytes() == pr.tobytes()
            and vertices[:len(pv)].tobytes() == pv.tobytes()
            and indices[:len(pi)].tobytes() == pi.tobytes(),
            'prior 599 surfaces changed')
    require(manifest['payload']['sha256'] == human.sha256(payload), 'new payload hash')
    source_pose = human.read_json(pose_path)
    torso = next(b for b in source_pose['bodies'] if b['body_index'] == 20)
    body_by_id = {b['body_index']: b for b in source_pose['bodies']}
    origin = np.asarray(torso['position_world_m'], float)
    left_axis = laterality.quaternion_rotation(torso['orientation_world_xyzw']) @ np.array([0., 0., -1.])
    require(len(manifest['candidates']) == len(SPECS), 'candidate count')
    expected_vstart, expected_istart = len(pv), len(pi)
    for row, (expected_source_id, expected_new_id, expected_member, expected_excluded) in zip(
            manifest['candidates'], SPECS, strict=True):
        require(row['source_stable_id'] == expected_source_id
                and row['candidate_stable_id'] == expected_new_id
                and row['source_member_id'] == expected_member
                and row['excluded_original_source_face_ids'] == list(expected_excluded),
                'exclusion identity/source face set changed')
        source = pr[row['source_stable_id']-1];candidate = records[row['candidate_stable_id']-1]
        fv,nv,fi,ni = map(int,[source[1],source[2],source[3],source[4]])
        parent_vertices = pv[fv:fv+nv]
        parent_faces = (pi[fi:fi+ni].reshape(-1,3)-fv)
        body = body_by_id[int(source[0])]
        original_world = parent_vertices[:,:3].astype(float) @ laterality.quaternion_rotation(
            body['orientation_world_xyzw']).T + body['position_world_m']
        original_support = laterality.support_metrics(original_world,parent_faces,origin,left_axis,-1,
                                                       list(range(len(parent_faces))))
        require(original_support['wrong_side_selected_face_ids'] == list(expected_excluded)
                and original_support['centroid_on_expected_side']
                and original_support['wrong_side_triangle_area_fraction'] == row['removed_surface_area_fraction'],
                'excluded source faces are not precisely the contralateral support')
        excluded = set(row['excluded_original_source_face_ids'])
        components = [set(group) for group in analyze_topology(
            parent_vertices[:,:3].astype(float).tolist(),parent_faces.tolist())['face_components']]
        require(all(group <= excluded or group.isdisjoint(excluded) for group in components),
                'excluded faces are not complete separate source components')
        require(sum(group <= excluded for group in components) == row['excluded_edge_component_count'],
                'excluded component count drift')
        kept_ids = [j for j in range(len(parent_faces)) if j not in excluded]
        kept = parent_faces[kept_ids]
        wrong_vertices = set(parent_faces[list(excluded)].ravel().tolist())
        right_vertices = set(kept.ravel().tolist())
        require(wrong_vertices.isdisjoint(right_vertices)
                and {tuple(parent_vertices[j,:3]) for j in wrong_vertices}.isdisjoint(
                    {tuple(parent_vertices[j,:3]) for j in right_vertices}),
                'excluded component shares native or spatial source vertices')
        used = sorted(set(kept.ravel().tolist()));mapping = {old:new for new,old in enumerate(used)}
        require(hashlib.sha256(np.asarray(used,dtype='<u4').tobytes()).hexdigest() ==
                row['retained_vertex_ids_sha256']
                and hashlib.sha256(np.asarray(kept_ids,dtype='<u4').tobytes()).hexdigest() ==
                row['retained_face_ids_sha256'],
                'retained source ancestry digest drift')
        vstart,vcount,istart,icount = map(int,[candidate[1],candidate[2],candidate[3],candidate[4]])
        actual_vertices = vertices[vstart:vstart+vcount]
        actual_faces = (indices[istart:istart+icount].reshape(-1,3)-vstart)
        expected_faces = np.asarray([[mapping[j] for j in f] for f in kept], dtype=np.uint32)
        require(vstart == expected_vstart and istart == expected_istart
                and candidate.tolist() == [int(source[0]),vstart,len(used),istart,len(kept)*3,
                                      row['candidate_stable_id'],14,0]
                and np.array_equal(actual_vertices,parent_vertices[used])
                and np.array_equal(actual_faces,expected_faces),
                'candidate/source face ancestry or owner differs')
        expected_vstart += len(used); expected_istart += len(kept)*3
        world = actual_vertices[:,:3].astype(float) @ laterality.quaternion_rotation(
            body['orientation_world_xyzw']).T + body['position_world_m']
        placed = laterality.support_metrics(world, actual_faces, origin, left_axis, -1, kept_ids)
        require(placed['all_selected_vertices_on_expected_side'], 'appended right-eye support wrong side')
        quotient = classify_quotient(actual_vertices[:,:3].astype(float).tolist(),actual_faces.tolist())
        require(quotient['status'] == row['compiled_quotient_status_after']
                and quotient['exact_intersection_pairs'] == row['compiled_quotient_exact_intersection_pairs_after']
                and quotient['topology']['closed_oriented_manifold_candidate'] == row['compiled_quotient_closed_after']
                and not quotient['closed_embedded_surface_candidate'],
                'source geometry failure was hidden or changed')
    require(expected_vstart == len(vertices) and expected_istart == len(indices),
            'appended candidate ranges incomplete')
    return {'schema': SCHEMA+'.audit', 'passed': True,
            'payload_sha256': human.sha256(payload),
            'all_first_599_surfaces_retained_byte_identically': True,
            'three_right_eye_laterality_candidates': [600,601,602],
            'three_geometry_failures_remain': True,
            'clinical_anatomy': False, 'physical_volume': False, 'mechanics': False}


def audit_native_selected(base, indexed, quotient, parent_598, source_proof,
                          selection_598, native_598_dir, pose_598,
                          parent_599, pack_599, pose_599,
                          new_payload, new_pack, new_pose_path):
    prior = choroid.audit_native_selected(base,indexed,quotient,parent_598,
        source_proof,selection_598,native_598_dir,pose_598,
        parent_599,pack_599,pose_599)
    require(prior['passed'] and prior['full_support_pass_after'] == 55
            and prior['remaining_wrong_side_source_ids'] == [413,426,427],
            '599-surface parent selected audit changed')
    manifest = human.read_json(Path(new_payload).with_name(MANIFEST_NAME))
    source_audit = audit_output(parent_599,pose_599,new_payload,manifest)
    pose_old = human.read_json(pose_599)
    pose_new = human.read_json(new_pose_path)
    require(pose_old['bodies'] == pose_new['bodies'], 'native source-rest body poses changed')
    old_selection = human.read_json(selection_598)
    hidden = sorted([*old_selection['hidden_anatomy_stable_ids'],587,413,426,427])
    native_audit = topology_audit.audit_native(new_payload,new_pack,new_pose_path,32767,hidden)
    gate = geometry_gate.audit(base,indexed,quotient,parent_598,
                               source_proof,selection_598,native_598_dir)
    selected = [dict(row) for row in gate['rows']]
    for source_id, visible_id in ((412,599),(413,600),(426,601),(427,602)):
        require(selected[source_id-1]['source_stable_id'] == source_id,
                'source/selected row mismatch')
        selected[source_id-1]['visible_stable_id'] = visible_id
    metadata,_ = laterality.source_metadata(Path(base).resolve())
    require(all(metadata[sid-1]['member_id'] == member for sid,_,member,_ in SPECS),
            'pinned source member identities changed')
    rows = laterality.inspect_selected(new_payload,selected,metadata,pose_new)
    require(len(rows) == 58 and all(r['centroid_on_expected_side']
            and r['all_selected_vertices_on_expected_side'] for r in rows),
            'selected bilateral placement still has wrong-side support')
    require([r['source_stable_id'] for r in rows if not r['geometry_embedded_candidate']]
            == [305,306,307,308,309,391,413,426,427],
            'bilateral source geometry failures changed')
    # The three new copies fix spatial placement only. Their geometry failure
    # remains asserted from fresh exact quotient audits above.
    by_id = {r['source_stable_id']: r for r in rows}
    require(all(not by_id[sid]['geometry_embedded_candidate'] for sid in (413,426,427))
            and all(by_id[sid]['visible_stable_id'] == visible for sid,visible in
                    ((412,599),(413,600),(426,601),(427,602))),
            'source geometry qualification or inspection selector changed')
    return {
        'schema':SCHEMA+'.native-selected-audit','passed':True,
        'source_audit':source_audit,'native_audit':native_audit,
        'source_rest_body_pose_unchanged':True,
        'bilateral_source_surface_count':58,
        'full_support_pass_before':55,'full_support_pass_after':58,
        'remaining_wrong_side_source_ids':[],
        'selected_right_eye_ids':{'412':599,'413':600,'426':601,'427':602},
        'remaining_geometry_failed_right_eye_source_ids':[413,426,427],
        'all_602_surfaces_retained_in_native_packet':True,
        'unassessed_source_surface_count':521,
        'clinical_anatomy':False,'physical_volume':False,'mechanics':False,
        'boundary':'All 58 explicitly bilateral surfaces pass full support laterality in one native source-rest packet. Three right-eye source meshes still fail geometry, 521 other source surfaces are not checked for laterality, and clinical anatomy and mechanics remain unqualified.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-payload', type=Path, required=True)
    parser.add_argument('--pose', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = compose(args.parent_payload, args.pose, args.output)
    proof = audit_output(args.parent_payload,args.pose,args.output/PAYLOAD_NAME,manifest)
    human.write_json(args.output/'right-eye-laterality-audit.json',proof)
    print(json.dumps({'payload_sha256':manifest['payload']['sha256'],
        'appended_candidate_ids':[r['candidate_stable_id'] for r in manifest['candidates']],
        'remaining_geometry_statuses':[r['compiled_quotient_status_after'] for r in manifest['candidates']]}),flush=True)


if __name__ == '__main__':
    main()
