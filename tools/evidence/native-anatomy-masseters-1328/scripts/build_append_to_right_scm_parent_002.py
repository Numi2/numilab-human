from pathlib import Path
import hashlib, json, math, shutil, struct, sys
import numpy as np

R = Path('/Users/n/numi-human-retained-delivery-20261009')
BASE = R/'head-muscle-coverage-1291/alternate-head-source-inventory-001/z-anatomy-range'
REG_DIR = BASE/'head-bone-triangle-registration-001'
HARM = REG_DIR/'masseter-harmonic-patch-001'
OUT = HARM/'append-to-right-scm-parent-002'
if OUT.exists():
    raise SystemExit(f'refuse existing output: {OUT}')
PARENT_DIR = R/'anatomy-completion-1276/right-scm-native-composition-001'
PARENT = PARENT_DIR/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
PARENT_MANIFEST = PARENT.with_suffix('.manifest.json')
PARENT_RECEIPT = PARENT_DIR/'resting-anatomy-receipt.json'
CAND = HARM/'bilateral-superficial-masseter-harmonic-source-trial.npz'
HARM_REPORT = HARM/'source-report.json'
HARM_RUNNER = HARM/'run_harmonic_source.py'
REG = REG_DIR/'registration-and-interface-report-003.json'
SOURCE_NPZ = BASE/'head-main-meshes.npz'
SOURCE_META = BASE/'head-main-meshes.json.meta'
SOURCE_NAMES = BASE/'head-main-meshes.json'
MARKERS_NPZ = BASE/'head-attachment-objects.npz'
MARKERS_META = BASE/'head-attachment-objects.meta'
BLEND = BASE/'Startup.blend'
LICENSE = BASE/'License.txt'
README = BASE/'Readme.md'
COMPOSER_ROOT = Path('/Users/n/numi-human-anatomy-publication-1308')
sys.path.insert(0, str(COMPOSER_ROOT/'src'))
from numilab_human import passive_attachment_composition as pac
from numilab_human import common_atlas_skin_clearance as ca

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = [PARENT, PARENT_MANIFEST, PARENT_RECEIPT, CAND, HARM_REPORT, HARM_RUNNER,
          REG, SOURCE_NPZ, SOURCE_META, SOURCE_NAMES, MARKERS_NPZ, MARKERS_META,
          BLEND, LICENSE, README,
          COMPOSER_ROOT/'src/numilab_human/passive_attachment_composition.py']
pins_before = {str(p): sha(p) for p in inputs}
parent = pac._read_nhtiss4(PARENT)
parent_manifest = json.loads(PARENT_MANIFEST.read_text())
parent_receipt = json.loads(PARENT_RECEIPT.read_text())
if parent['surface_count'] != 151 or parent_manifest['payload']['sha256'] != pins_before[str(PARENT)]:
    raise RuntimeError('unexpected immediate parent identity or row count')
if parent_receipt['provenance']['native_muscle_surfaces']['sha256'] != pins_before[str(PARENT)]:
    raise RuntimeError('parent anatomy receipt is not bound to the 151-row asset')
record151 = next(r for r in parent['records'] if int(r[6]) == 151)
local151 = pac._row_slices(parent, record151)
head_bindings = []
for i in range(local151['binding_count']):
    raw = local151['binding_bytes'][i*36:(i+1)*36]
    decoded = struct.unpack('<I8f', raw)
    if decoded[0] == 23:
        head_bindings.append((raw, decoded))
if len(head_bindings) != 1:
    raise RuntimeError('row 151 does not have exactly one core-23 static binding')
head_binding_bytes, hb = head_bindings[0]

source_report = json.loads(HARM_REPORT.read_text())
registration_report = json.loads(REG.read_text())
registration = registration_report['source_registration']
mat = np.asarray(registration['head_refined_source_to_bodyparts_matrix'], dtype=np.float64)
head = registration['head_binding']
head_rot = np.asarray(head['rotation_matrix'], dtype=np.float64)
head_trans = np.asarray(head['translation_m'], dtype=np.float64)
head_scale = float(head['scale'])
parent_translation = np.asarray(hb[1:4], dtype=np.float64)
parent_quat = np.asarray(hb[4:8], dtype=np.float64)
parent_scale = float(hb[8])
parent_rotation = ca._rotation(parent_quat)
if hb[0] != 23 or np.max(np.abs(parent_translation-head_trans)) > 2e-6 \
        or abs(parent_scale-head_scale) > 2e-6 \
        or np.max(np.abs(parent_rotation-head_rot)) > 2e-6:
    raise RuntimeError('parent core-23 binding disagrees with the source registration report')

meta = json.loads(SOURCE_META.read_text())
objects = {o['object_name']: o for o in meta['objects']}
original = np.load(SOURCE_NPZ, allow_pickle=False)
candidate = np.load(CAND, allow_pickle=False)
faces_by_id = {159: None, 160: None}
rows, bindings, vertex_chunks, face_chunks = [], [], [], []
row_geometry = {}
vertex_offset = index_offset = binding_offset = 0
row_specs = [
    (159, 'FMA49001', 'right superficial masseter', 'Superficial part of masseter.r'),
    (160, 'FMA49002', 'left superficial masseter', 'Superficial part of masseter.l'),
]
for stable_id, member_id, source_name, object_name in row_specs:
    key = object_name.replace(' ', '_')
    z_vertices = np.asarray(candidate[key+'__vertices_f32'], dtype='<f4')
    faces = np.asarray(candidate[key+'__faces_i32'], dtype=np.int64)
    ancestry = np.asarray(candidate[key+'__face_origins_i32'], dtype=np.int64)
    if z_vertices.shape != (702, 3) or faces.shape != (1404, 3) \
            or not np.array_equal(ancestry, np.arange(len(faces), dtype=np.int64)):
        raise RuntimeError(f'{object_name}: candidate shape/face ancestry mismatch')
    obj_meta = objects[object_name]
    original_vertices = np.asarray(original[obj_meta['prefix']+'_world_vertices_f64'], dtype='<f4')
    original_faces = np.asarray(original[obj_meta['prefix']+'_triangles_i32'], dtype=np.int64)
    if original_vertices.shape != z_vertices.shape or not np.array_equal(original_faces, faces):
        raise RuntimeError(f'{object_name}: source face stream changed')
    corrected_source_delta = (z_vertices.astype(np.float64)-original_vertices.astype(np.float64))
    bp = z_vertices.astype(np.float64) @ mat[:3, :3].T + mat[:3, 3]
    bp32 = np.asarray(bp, dtype='<f4')
    face_vec = bp[faces]
    face_normals = np.cross(face_vec[:, 1]-face_vec[:, 0], face_vec[:, 2]-face_vec[:, 0])
    accum = np.zeros_like(bp, dtype=np.float64)
    for j in range(3):
        np.add.at(accum, faces[:, j], face_normals)
    lengths = np.linalg.norm(accum, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-20):
        raise RuntimeError(f'{object_name}: no area-weighted normal for one or more vertices')
    normals32 = np.asarray(accum/lengths[:, None], dtype='<f4')
    # Recompute source-frame proxy positions under the existing parent static
    # head binding; this is the exact coordinate path used by the pose audit.
    canonical = parent_translation + parent_scale*(bp32.astype(np.float64) @ parent_rotation.T)
    expected_canonical = head_trans + head_scale*(bp32.astype(np.float64) @ head_rot.T)
    if np.max(np.abs(canonical-expected_canonical)) > 1e-7:
        raise RuntimeError(f'{object_name}: packed row binding does not reproduce the audited head-local coordinates')
    row_v_bytes = b''.join(
        struct.pack('<6f4I4f', *bp32[i], *normals32[i],
                    0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                    1.0, 0.0, 0.0, 0.0)
        for i in range(len(bp32))
    )
    face_bytes = np.asarray(faces, dtype='<u4').tobytes(order='C')
    rows.append(struct.pack('<8I', binding_offset, 1, vertex_offset, len(bp32),
                            index_offset, faces.size, stable_id, 1))
    bindings.append(head_binding_bytes)
    vertex_chunks.append(row_v_bytes)
    # NHTISS subset indices address the concatenated subset vertex array.
    face_chunks.append(np.asarray(faces + vertex_offset, dtype='<u4').tobytes(order='C'))
    row_geometry[stable_id] = {
        'stable_id': stable_id, 'member_id': member_id, 'source_name': source_name,
        'source_object_name': object_name,
        'vertex_count': len(bp32), 'triangle_count': len(faces),
        'source_world_bounds_m': {'min': z_vertices.min(0).astype(float).tolist(), 'max': z_vertices.max(0).astype(float).tolist()},
        'registered_bodyparts_world_bounds_m': {'min': bp32.min(0).astype(float).tolist(), 'max': bp32.max(0).astype(float).tolist()},
        'head_body_local_bounds_m': {'min': canonical.min(0).astype(float).tolist(), 'max': canonical.max(0).astype(float).tolist()},
        'max_source_correction_from_original_m': float(np.linalg.norm(corrected_source_delta, axis=1).max()),
        'changed_source_vertex_count': int(np.count_nonzero(np.any(corrected_source_delta != 0.0, axis=1))),
        'max_registered_correction_from_original_m': float(np.linalg.norm((z_vertices.astype(float)-original_vertices.astype(float)) @ mat[:3,:3].T, axis=1).max()),
        'min_weight_core_23': 1.0, 'max_weight_core_23': 1.0,
        'other_core_weight_min_max': [0.0, 0.0],
        'one_hot_head_support_vertex_count': len(bp32),
        'attachment_masks_supplied': False,
        'source_faces_preserved': True,
        'source_face_origins': {'method': 'identity_original_triangle_row', 'first': int(ancestry.min()), 'last': int(ancestry.max())},
        'source_original_candidate_vertex_arrays_sha256': hashlib.sha256(original_vertices.tobytes()+faces.astype('<i4').tobytes()).hexdigest(),
        'registered_candidate_vertex_arrays_sha256': hashlib.sha256(bp32.tobytes()+faces.astype('<i4').tobytes()).hexdigest(),
    }
    vertex_offset += len(bp32)
    index_offset += faces.size
    binding_offset += 1

header = struct.pack('<8s6I32s', b'NHTISS4\0', 5, len(rows), len(bindings),
                     vertex_offset, index_offset, parent['fingerprint'], parent['source_digest'])
subset_raw = header+b''.join(rows+bindings+vertex_chunks+face_chunks)
OUT.mkdir(parents=True, exist_ok=False)
subset_dir = OUT/'subset'; subset_dir.mkdir()
subset_payload = subset_dir/'head-masseter-passive-subset.nhtissue'
subset_payload.write_bytes(subset_raw)
subset_rows = []
for item in row_geometry.values():
    sid = item['stable_id']
    zmarker = 'Masseter.er' if sid == 159 else 'Masseter.el'
    origin_marker = 'Masseter.o1r' if sid == 159 else 'Masseter.o1l'
    marker_report = registration_report['source_attachment_marker_to_canonical_bone_triangle_distances']
    subset_rows.append({
        'stable_id': sid, 'member_id': item['member_id'], 'source_name': item['source_name'],
        'layer': 'muscle', 'vertex_count': item['vertex_count'], 'triangle_count': item['triangle_count'],
        'matched_muscles': [],
        'passive_visual_binding': {'method': 'registered_head_rigid_visual_support', 'supports': ['head'],
            'support_core_body_index': 23, 'mechanics_changed': False, 'myosim_route_added': False,
            'jaw_function': 'not_modeled', 'inferred_support_not_source_attachment_mask': True},
        'external_source': {
            'atlas': 'Z-Anatomy', 'source_object_name': item['source_object_name'],
            'source_member_id': item['member_id'],
            'source_array_export_path': str(SOURCE_NPZ), 'source_array_export_sha256': pins_before[str(SOURCE_NPZ)],
            'source_export_metadata_path': str(SOURCE_META), 'source_export_metadata_sha256': pins_before[str(SOURCE_META)],
            'source_object_inventory_path': str(SOURCE_NAMES), 'source_object_inventory_sha256': pins_before[str(SOURCE_NAMES)],
            'source_blender_member': 'Startup.blend', 'source_blender_member_path': str(BLEND),
            'source_blender_member_sha256': pins_before[str(BLEND)],
            'license': 'CC BY-SA 4.0 International',
            'required_attribution': 'Z-Anatomy - The libre 3D atlas of anatomy - CC BY-SA 4.0',
            'upstream_attribution': 'BodyParts3D - The Database Center for Life Science - CC BY-SA 2.1 Japan',
            'license_text_path': str(LICENSE), 'license_text_sha256': pins_before[str(LICENSE)],
            'readme_path': str(README), 'readme_sha256': pins_before[str(README)],
            'registration_report_path': str(REG), 'registration_report_sha256': pins_before[str(REG)],
            'harmonic_source_report_path': str(HARM_REPORT), 'harmonic_source_report_sha256': pins_before[str(HARM_REPORT)],
            'source_correction': 'radius-1 uniform-Laplacian patch, blend alpha 0.25; inferred reference cleanup, not a measured individual',
            'source_attachment_proxy_markers': {
                'insertion': marker_report[zmarker], 'origin_proxy': marker_report[origin_marker],
                'source_marker_object_file': str(MARKERS_NPZ), 'source_marker_object_sha256': pins_before[str(MARKERS_NPZ)],
                'source_marker_metadata_file': str(MARKERS_META), 'source_marker_metadata_sha256': pins_before[str(MARKERS_META)],
                'interpretation': 'Unchanged Blender endpoint/attachment-proxy objects and their bone distances; no vertex attachment footprint mask is supplied or inferred.'
            }
        },
        'source_geometry_bounds': item,
    })
subset_manifest = {
    'schema': 'numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1',
    'payload': {'file': subset_payload.name, 'sha256': hashlib.sha256(subset_raw).hexdigest(),
        'bytes': len(subset_raw), 'surface_count': 2, 'binding_count': 2,
        'vertex_count': vertex_offset, 'index_count': index_offset,
        'registration_fingerprint32': f'{parent["fingerprint"]:08x}'},
    'source': {
        'bodyparts': parent_manifest['source']['bodyparts'],
        'myosim_source_archive_sha256': parent_manifest['source']['myosim_source_archive_sha256'],
        'registration': parent_manifest['source']['registration'],
        'myosim_manifest': parent_manifest['source']['myosim_manifest'],
        'surfaces': subset_rows,
        'external_source_attribution': ['Z-Anatomy CC BY-SA 4.0', 'BodyParts3D - DBCLS CC BY-SA 2.1 Japan'],
        'geometry_interpretation': 'Two source-registered inferred passive head surfaces; they add no MyoSim route or jaw function.'
    },
    'coverage': {'configured_surface_count': 2, 'emitted_surface_count': 2,
        'selected_stable_ids': [159,160], 'muscle_surface_count': 2, 'tendon_surface_count': 0,
        'route_bound_muscle_surface_count': 0, 'passive_registered_support_surface_count': 2},
    'runtime_binding': 'One exact existing Core body-23 registered head transform per row; one-hot head-support visual weights only.',
    'status': 'inferred_passive_head_reference_visual_input_not_collision_or_physics',
    'evidence_boundary': 'Source/pose/target geometry is checked offline. This does not establish measured attachments, jaw mechanics, anatomy admission, or native cycle admission.'
}
subset_manifest_path = subset_payload.with_suffix('.manifest.json')
subset_manifest_path.write_text(json.dumps(subset_manifest, sort_keys=True, indent=2)+'\n')

proof = pac.append_passive_surfaces(PARENT, subset_payload, OUT/'assets', stable_ids=(159,160))
child = OUT/'assets'/PARENT.name
bound_receipt = pac.bind_anatomy_receipt(PARENT_RECEIPT, child, OUT/'resting-anatomy-receipt.json')
if bound_receipt['provenance']['native_muscle_surfaces']['sha256'] != sha(child) \
        or bound_receipt['provenance']['native_muscle_surfaces']['surface_count'] != 153:
    raise RuntimeError('bound receipt does not identify the 153-row child')

# Retain exact license text and upstream README alongside the row provenance.
rights = OUT/'source-rights'; rights.mkdir()
shutil.copyfile(LICENSE, rights/'Z-Anatomy-License.txt')
shutil.copyfile(README, rights/'Z-Anatomy-Readme.md')
if sha(rights/'Z-Anatomy-License.txt') != pins_before[str(LICENSE)] or sha(rights/'Z-Anatomy-Readme.md') != pins_before[str(README)]:
    raise RuntimeError('retained attribution text changed during copy')

# Compact registration residual evidence; the mandible is held out from the fit.
residuals = {}
for bone, stats in registration['head_bone_sample_metrics_after_residual'].items():
    d = stats.get('source_to_raw_nhbones_triangle_distance_mm', {})
    residuals[bone] = {k: d.get(k) for k in ('count','min_mm','median_mm','p95_mm','p99_mm','max_mm')}
val_members = registration.get('validation_members', [])
registration_summary = {
    'source_frame': registration.get('source_mesh_frame'),
    'target_frame': registration.get('target_frame'),
    'thorax_prior_report_sha256': registration.get('thorax_derived_prior_sha256'),
    'residual_rotation_deg': registration.get('head_residual_rotation_angle_deg'),
    'refined_scale': registration.get('head_refined_scale'),
    'fit_method': registration.get('validation_method'),
    'fit_and_heldout_members': val_members,
    'bone_sample_triangle_distance_mm': residuals,
    'head_binding_core_index': 23,
    'head_binding_static_table_matches_parent_row151': True,
}
report = {
    'schema': 'numi.zanatomy-superficial-masseter-passive-append-preparation.v1',
    'status': 'composed_offline_unadmitted', 'complete': True, 'native_admission': False,
    'parent_payload_path': str(PARENT), 'parent_payload_sha256': pins_before[str(PARENT)],
    'parent_manifest_path': str(PARENT_MANIFEST), 'parent_manifest_sha256': pins_before[str(PARENT_MANIFEST)],
    'parent_surface_count': 151,
    'subset_payload_path': str(subset_payload), 'subset_payload_sha256': sha(subset_payload),
    'subset_manifest_path': str(subset_manifest_path), 'subset_manifest_sha256': sha(subset_manifest_path),
    'appended_payload_path': str(child), 'appended_payload_sha256': sha(child),
    'appended_manifest_path': str(child.with_suffix('.manifest.json')), 'appended_manifest_sha256': sha(child.with_suffix('.manifest.json')),
    'bound_receipt_path': str(OUT/'resting-anatomy-receipt.json'), 'bound_receipt_sha256': sha(OUT/'resting-anatomy-receipt.json'),
    'appended_stable_ids': [159,160], 'parent_rows_and_bindings_byte_exact': proof['parent_rows_and_bindings_byte_exact'],
    'rows': list(row_geometry.values()),
    'registered_bone_fit_residuals': registration_summary,
    'attachment_proxy_bounds': {
        str(r['stable_id']): r['external_source']['source_attachment_proxy_markers']
        for r in subset_rows
    },
    'no_attachment_mask_or_jaw_mechanics_claim': True,
    'source_attribution_and_license': {
        'license': 'Z-Anatomy CC BY-SA 4.0 International; upstream BodyParts3D DBCLS CC BY-SA 2.1 Japan attribution retained.',
        'license_copy_path': str(rights/'Z-Anatomy-License.txt'), 'license_sha256': sha(rights/'Z-Anatomy-License.txt'),
        'readme_copy_path': str(rights/'Z-Anatomy-Readme.md'), 'readme_sha256': sha(rights/'Z-Anatomy-Readme.md')
    },
    'composition_proof': proof,
    'input_sha256_before': pins_before,
}
report_path = OUT/'append-preparation-report.json'
report_path.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
pins_after = {str(p): sha(p) for p in inputs}
if pins_before != pins_after:
    raise RuntimeError('an input changed during row construction/composition')
report['input_sha256_after'] = pins_after
report['inputs_unchanged'] = True
report_path.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
print(json.dumps({'status':report['status'],'parent_sha256':pins_before[str(PARENT)],
 'subset_payload_sha256':report['subset_payload_sha256'],'appended_payload_sha256':report['appended_payload_sha256'],
 'appended_manifest_sha256':report['appended_manifest_sha256'],'bound_receipt_sha256':report['bound_receipt_sha256'],
 'surface_count':153,'rows':[{k:r[k] for k in ('stable_id','member_id','source_name','vertex_count','triangle_count','max_source_correction_from_original_m','changed_source_vertex_count','one_hot_head_support_vertex_count')} for r in report['rows']],
 'inputs_unchanged':True,'report_path':str(report_path),'report_sha256':sha(report_path)},sort_keys=True),flush=True)
