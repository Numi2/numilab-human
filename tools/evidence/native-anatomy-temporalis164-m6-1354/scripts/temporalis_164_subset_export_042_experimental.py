from pathlib import Path
import sys, hashlib, json, struct, copy, collections
import numpy as np
W=Path('/Users/n/numi-human-exact-normalization-m6-1348')
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
P=Path('/Users/n/numi-human-complete-reference-inventory-1335')
H=Path('/Users/n/numi-human-retained-delivery-20261009/head-muscle-coverage-1291/alternate-head-source-inventory-001/z-anatomy-range')
OUT=W/'reports/temporalis-clearance-002/passive-subset-042-experimental/run-001/result-001'
SUB=OUT/'subset'; NAME='bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
C=W/'reports/temporalis-clearance-002/native-witness-repair-164-042/run-001/result-001/temporalis-164-left-skin-clearance-candidate-041.npz'
RP=W/'reports/temporalis-clearance-002/native-witness-repair-164-042/run-001/result-001/report.json'
PARENT=A/'temporalis163-native-m6-001/assets'/NAME; PM=PARENT.with_suffix('.manifest.json')
ADAPTER=P/'src/numilab_human/passive_attachment_composition.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(ok,msg):
    if not ok: raise RuntimeError(msg)
sys.path.insert(0,str(P/'src'))
from numilab_human import passive_attachment_composition as pc
witness=json.loads(RP.read_text())
req(witness.get('status')=='native_capture_local_repair_passed' and witness.get('whole25_anatomy_and_external_clearance_not_yet_verified') is True,'local native witness report is not the expected experimental receipt')
for p,d in witness['input_sha256'].items(): req(Path(p).is_file() and sha(p)==d,'local witness input pin mismatch: '+p)
req(Path(witness['candidate041_path']).resolve()==C.resolve() and sha(C)==witness['candidate041_sha256'],'candidate041 path/hash mismatch')
ROUNDTRIP=W/'reports/temporalis-clearance-002/skin-clip-clearance-164-039/run-002/result-001/report.json'
proof=json.loads(ROUNDTRIP.read_text())
C040=W/'reports/temporalis-clearance-002/skin-clip-clearance-164-039/run-002/result-001/temporalis-164-left-skin-clearance-candidate-040.npz'
req(proof.get('status')=='candidate_clear_all25' and proof.get('all25_captures_clear') is True and proof.get('all25_captures')==25 and proof.get('stable_id')==164,'candidate040 baseline all25 report changed')
for p,d in proof['input_sha256'].items(): req(Path(p).is_file() and sha(p)==d,'candidate040 baseline pin mismatch: '+p)
req(Path(proof['candidate_npz']).resolve()==C040.resolve() and sha(C040)==proof['candidate_npz_sha256'],'candidate040 baseline path/hash mismatch')
req(PARENT.is_file() and PM.is_file() and sha(PM)=='cb4374524b3a65cd881d4f3655dada323d9d26e65f427d39f52d627e3550b726','current append parent changed')
parent=pc._read_nhtiss4(PARENT); parent_manifest=json.loads(PM.read_text())
req(parent['surface_count']==157 and 164 not in [int(r[6]) for r in parent['records']] and 163 in [int(r[6]) for r in parent['records']],'parent inventory mismatch')
row159=next(r for r in parent['records'] if int(r[6])==159); local159=pc._row_slices(parent,row159)
req(local159['binding_count']==1 and struct.unpack_from('<I',local159['binding_bytes'])[0]==23,'row159 is not core23 binding')
with np.load(C,allow_pickle=False) as z:
    v6=np.asarray(z['vertices6'],dtype='<f4'); bi=np.asarray(z['binding_indices'],dtype='<u4'); weights=np.asarray(z['weights'],dtype='<f4')
    faces=np.asarray(z['faces'],dtype='<u4'); ancestry=json.loads(str(z['face_ancestry_json'].item()))
    bind_sha=str(z['binding_record_sha256'][0]); stable=int(np.asarray(z['stable_id'])[0]); world=np.asarray(z['step0_world_f32'],dtype='<f4')
req(stable==164 and v6.shape==(5063,6) and faces.shape==(10134,3) and len(ancestry)==len(faces),'candidate 035 arrays changed')
req(hashlib.sha256(local159['binding_bytes']).hexdigest()==bind_sha,'candidate binding differs from parent row159')
req(bi.shape==(len(v6),4) and np.all(bi[:,0]==0) and np.all(bi[:,1:]==0xffffffff) and np.all(weights[:,0]==1.0) and np.all(weights[:,1:]==0.0),'candidate not one-hot core23')
req(faces.min()>=0 and faces.max()<len(v6),'face index outside vertex allocation')
face_counts=dict(collections.Counter(str(x.get('kind','unknown')) for x in ancestry))
C0=W/'reports/temporalis-clearance-002/skin-clip-clearance-164-039/run-001/result-001/temporalis-164-left-skin-clearance-candidate-039.npz'
TARGET=W/'reports/temporalis-clearance-002/skin-cap-center-audit-017/run-001/result-001/temporary-classifier-target.npz'
bone_report_path=W/'reports/temporalis-clearance-002/bone-clearance-164-037/run-006/result-001/bone-clearance-report.json'
bone_report=json.loads(bone_report_path.read_text())
diag_path=Path(bone_report['source_diagnostic_path']); diag_hash=sha(diag_path)
req(bone_report['stable_id']==164 and bone_report['member_id']=='FMA49008' and bone_report['source_object_name']=='Temporalis muscle.l' and diag_hash==bone_report['source_diagnostic_sha256'],'left source diagnostic identity/hash mismatch')
with np.load(diag_path,allow_pickle=False) as dz:
    source_vertices_before=int(len(dz['vertices6'])); source_faces_before=int(len(dz['faces']))
req(source_vertices_before==5126 and source_faces_before==10260,'left diagnostic source counts changed')
with np.load(C040,allow_pickle=False) as zf:
    final_source=np.asarray(zf['positions_m'],dtype='<f4')
    final_world=np.asarray(zf['step0_world_f32'],dtype='<f4')
with np.load(C,allow_pickle=False) as zw:
    witness_source=np.asarray(zw['positions_m'],dtype='<f4')
with np.load(C0,allow_pickle=False) as zb:
    base_source=np.asarray(zb['positions_m'],dtype='<f4')
    base_world=np.asarray(zb['step0_world_f32'],dtype='<f4')
    base_faces=np.asarray(zb['faces'],dtype='<u4')
req(np.array_equal(faces,base_faces),'pre/post correction face topology differs')
source_delta=final_source.astype(np.float64)-base_source.astype(np.float64)
changed=np.flatnonzero(np.any(source_delta!=0.0,axis=1))
req(changed.tolist()==[4417],'round-trip correction must be limited to vertex 4417')
world_delta=final_world[4417].astype(np.float64)-base_world[4417].astype(np.float64)
req(np.count_nonzero(final_world.astype(np.float64)-base_world.astype(np.float64))==3,'round-trip correction changed more than one stored world vertex')
patch_mask=np.asarray([str(x.get('kind',''))=='scalp_clipping_patch' for x in ancestry],dtype=bool)
patch_tri=final_world[faces].astype(np.float64)
patch_cross=np.cross(patch_tri[:,1]-patch_tri[:,0],patch_tri[:,2]-patch_tri[:,0])
patch_vertex_cross=np.zeros_like(final_world,dtype=np.float64)
for col in range(3): np.add.at(patch_vertex_cross,faces[patch_mask][:,col],patch_cross[patch_mask])
local_normal=patch_vertex_cross[4417]/np.linalg.norm(patch_vertex_cross[4417])
with np.load(TARGET,allow_pickle=False) as skin:
    skin_vertices=np.asarray(skin['vertices_f32'],dtype=np.float64)
    skin_faces=np.asarray(skin['faces_i64'],dtype=np.int64)
source_skin_ids=np.asarray([int(ancestry[i]['original_skin_quotient_face']) for i in np.flatnonzero(patch_mask)],dtype=np.int64)
skin_tri=skin_vertices[skin_faces[source_skin_ids]]
skin_cross=np.cross(skin_tri[:,1]-skin_tri[:,0],skin_tri[:,2]-skin_tri[:,0])
source_skin_normal=skin_cross.sum(axis=0); source_skin_normal/=np.linalg.norm(source_skin_normal)
roundtrip_projection_audit={
    'pre_correction_candidate_npz_path':str(C0),'pre_correction_candidate_npz_sha256':sha(C0),
    'corrected_candidate_npz_path':str(C040),'corrected_candidate_npz_sha256':sha(C040),
    'bone_clearance_report_path':str(bone_report_path),'bone_clearance_report_sha256':sha(bone_report_path),
    'left_source_diagnostic_path':str(diag_path),'left_source_diagnostic_sha256':diag_hash,
    'corrected_vertex_index':4417,'changed_source_vertex_indices':changed.astype(int).tolist(),
    'source_frame_delta_m':source_delta[4417].tolist(),'step0_world_delta_m':world_delta.tolist(),
    'step0_world_delta_norm_m':float(np.linalg.norm(world_delta)),
    'source_skin_normal_projection_definition':'Dot the corrected-minus-pre-correction Float32 step-0 world displacement with the unit normal formed by summing area vectors of the 770 mapped original skin quotient triangles, retaining one contribution per scalp clipping descendant face.',
    'source_skin_normal_projection_m':float(world_delta@source_skin_normal),
    'patch_vertex_normal_projection_definition':'Dot the corrected-minus-pre-correction Float32 step-0 world displacement with the unit area-weighted normal formed from corrected scalp-clipping child-face area vectors incident on vertex 4417.',
    'patch_vertex_normal_projection_m':float(world_delta@local_normal),
    'reported_clearance_worker_patch_vertex_projection_m':float(proof['local_roundtrip_separation_repair']['step0_skin_outward_component_m']),
    'normal_projection_values_are_distinct_definitions':True,
    'original_skin_quotient_face_count_with_descendant_multiplicity':int(len(source_skin_ids)),
    'temporary_skin_target_npz_path':str(TARGET),'temporary_skin_target_npz_sha256':sha(TARGET),
    'temporary_target_authoring_only':True,'runtime_skin_modified':False,'runtime_ocular_openings_changed':False}
req(abs(roundtrip_projection_audit['step0_world_delta_norm_m']-float(proof['local_roundtrip_separation_repair']['step0_world_delta_m']))<1e-12,'round-trip world correction differs from candidate040 baseline report')
witness_source_delta=witness_source.astype(np.float64)-final_source.astype(np.float64)
witness_changed=np.flatnonzero(np.any(witness_source_delta!=0.0,axis=1))
req(witness_changed.tolist()==[2119] and np.array_equal(faces,base_faces),'candidate041 witness edit is not isolated to vertex2119')
native_witness_repair={'report_path':str(RP),'report_sha256':sha(RP),'candidate040_sha256':sha(C040),'candidate041_sha256':sha(C),
    'operation':witness['repair']['operation'],'source_vertex_index':2119,'source_delta_m':witness_source_delta[2119].tolist(),
    'source_displacement_norm_m':float(np.linalg.norm(witness_source_delta[2119])),
    'nonshared_pair_min_gap_across_eight_actual_captures_m':witness['nonshared_pair_min_triangle_gap_across_eight_actual_native_captures_m'],
    'moved_vertex_minimum_bone32_gap_across_eight_actual_captures_m':witness['minimum_moved_vertex_bone32_gap_across_eight_actual_native_captures_m'],
    'eight_capture_local_self_and_bone32_checks_passed':True,'all25_screen_pending':True,'new_candidate_native_run_pending':True}
fp,sd=parent['fingerprint'],parent['source_digest']; nv=len(v6); ni=int(faces.size)
header=struct.pack('<8s6I32s',b'NHTISS4\0',5,1,1,nv,ni,fp,sd)
record=struct.pack('<8I',0,1,0,nv,0,ni,164,1)
vertex_blob=bytearray(nv*56)
np.ndarray((nv,6),'<f4',buffer=vertex_blob,offset=0,strides=(56,4))[:]=v6
np.ndarray((nv,4),'<u4',buffer=vertex_blob,offset=24,strides=(56,4))[:]=bi
np.ndarray((nv,4),'<f4',buffer=vertex_blob,offset=40,strides=(56,4))[:]=weights
payload=header+record+local159['binding_bytes']+bytes(vertex_blob)+faces.reshape(-1).astype('<u4').tobytes()
req(len(payload)==64+32+36+nv*56+ni*4,'NHTISS ABI byte count mismatch')
OUT.mkdir(parents=True,exist_ok=False); SUB.mkdir()
subpath=SUB/NAME; subpath.write_bytes(payload)
source_report=W/'reports/temporalis-clearance-002/run-001/result-001/report.json'
reg=H/'head-bone-triangle-registration-001/registration-and-interface-report-003.json'
source_npz=H/'head-main-meshes.npz'; inventory=H/'head-main-meshes.json'; meta=H/'head-main-meshes.json.meta'
blend=H/'Startup.blend'; readme=H/'Readme.md'; license=H/'License.txt'
original=json.loads(source_report.read_text()); markers=original.get('source_markers',{}); regdoc=json.loads(reg.read_text())
left_markers={k:v for k,v in markers.items() if k.endswith('.el') or k.endswith('.o1l') or k.endswith('.o2l')}
external={'atlas':'Z-Anatomy','license':'CC BY-SA 4.0 International','license_text_path':str(license),'license_text_sha256':sha(license),'readme_path':str(readme),'readme_sha256':sha(readme),'required_attribution':'Z-Anatomy - The libre 3D atlas of anatomy - CC BY-SA 4.0','source_blender_member':'Startup.blend','source_blender_member_path':str(blend),'source_blender_member_sha256':sha(blend),'source_array_export_path':str(source_npz),'source_array_export_sha256':sha(source_npz),'source_object_inventory_path':str(inventory),'source_object_inventory_sha256':sha(inventory),'source_export_metadata_path':str(meta),'source_export_metadata_sha256':sha(meta),'registration_report_path':str(reg),'registration_report_sha256':sha(reg),'source_member_id':'FMA49008','source_object_name':'Temporalis muscle.l','source_family':'Z-Anatomy Startup.blend original mesh data; FMA49008 is a semantic key, not a BP3D tessellation claim','source_attachment_proxy_markers':left_markers,'attachment_mask_supplied':False,'source_report_path':str(source_report),'source_report_sha256':sha(source_report),'source_report_scope':'Shared source marker inventory containing all left/right marker records; this report geometry row is FMA49007/right and is not used as left-source identity.','left_source_diagnostic_path':str(diag_path),'left_source_diagnostic_sha256':diag_hash,'bone_clearance_report_path':str(bone_report_path),'bone_clearance_report_sha256':sha(bone_report_path),'candidate_npz_path':str(C),'candidate_npz_sha256':sha(C),'clearance_report_path':str(RP),'clearance_report_sha256':sha(RP),'source_lift_transform':regdoc.get('source_registration'),'interpretation':'Experimental registered Z-Anatomy left temporalis reference geometry with exact bone-shell subtraction, scalp clipping and a locally witnessed 4um inferred bone-cut-boundary adjustment; passive visual support only, all25 screening and a new native run remain pending. Attachment markers are proximity proxies, not a measured footprint.'}
local_bounds={'min':v6[:,:3].min(axis=0).astype(float).tolist(),'max':v6[:,:3].max(axis=0).astype(float).tolist()}
world_bounds={'min':world.min(axis=0).astype(float).tolist(),'max':world.max(axis=0).astype(float).tolist()}
row={'stable_id':164,'member_id':'FMA49008','source_name':'left temporalis','layer':'muscle','matched_muscles':[],'vertex_count':nv,'triangle_count':len(faces),'external_source':external,'passive_visual_binding':{'inferred_support_not_source_attachment_mask':True,'method':'registered_head_rigid_visual_support','supports':['head'],'support_core_body_index':23,'mechanics_changed':False,'myosim_route_added':False,'jaw_function':'not_modeled'},'source_geometry_bounds':{'member_id':'FMA49008','source_object_name':'Temporalis muscle.l','head_body_local_bounds_m':local_bounds,'registered_bodyparts_world_bounds_m':world_bounds,'min_weight_core_23':1.0,'max_weight_core_23':1.0,'other_core_weight_min_max':[0.0,0.0],'one_hot_head_support_vertex_count':nv,'attachment_masks_supplied':False,'source_faces_preserved':False,'source_face_ancestry_counts':face_counts,'candidate_npz_sha256':sha(C),'source_diagnostic_sha256':diag_hash,'local_roundtrip_separation_repair':roundtrip_projection_audit,'native_witness_boundary_repair':native_witness_repair},'reference_surface_reconstruction':{'schema':'offline.temporalis.source-derived-reference-surface-reconstruction.v1','stable_id':164,'vertex_count_before':source_vertices_before,'vertex_count_after':nv,'triangle_count_before':source_faces_before,'triangle_count_after':len(faces),'candidate_npz_path':str(C),'candidate_npz_sha256':sha(C),'local_native_witness_report_path':str(RP),'local_native_witness_report_sha256':sha(RP),'original_source_object':'Temporalis muscle.l','original_source_member_id':'FMA49008','original_source_diagnostic_path':str(diag_path),'original_source_diagnostic_sha256':diag_hash,'original_source_vertices':source_vertices_before,'original_source_faces':source_faces_before,'method':'Exact arrangement subtraction against four registered bone-shell operands; exact clipping against temporary closed scalp authoring target; 15 micrometre inferred inward scalp reserve; one-vertex inverse-forward Float32 separation correction; then a bounded 4um source-frame move of bone-cut-boundary vertex2119 after a measured native Float32 crossing. The local witness receipt clears the repair across eight actual captures; all25 screening and a fresh native run remain pending.','bone_shell_cut_ancestry_preserved':True,'scalp_clipping_patch_ancestry_preserved':True,'source_face_ancestry_preserved_for_retained_faces':True,'face_ancestry_counts':face_counts,'face_ancestry_embedded_in_candidate_npz':True,'temporary_target_authoring_only':True,'runtime_skin_modified':False,'runtime_ocular_openings_changed':False,'local_roundtrip_separation_repair':roundtrip_projection_audit,'admission_state':'experimental_only','local_native_witness_receipt':'passed for eight actual captures; see pinned local receipt','all25_capture_gate':'pending; do not treat as admitted until all25 and new native verification pass'}}
manifest=copy.deepcopy(parent_manifest)
manifest['payload']={'file':NAME,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'surface_count':1,'binding_count':1,'vertex_count':nv,'index_count':ni,'registration_fingerprint32':f'{fp:08x}'}
manifest['source']['surfaces']=[row]
manifest['source']['temporalis_164_reference_surface_reconstruction']={'candidate_npz_sha256':sha(C),'local_native_witness_report_sha256':sha(RP),'all25_screen_path':'/Users/n/numi-human-exact-normalization-m6-1348/reports/temporalis-clearance-002/all25-candidate-screen-164-043/run-002/result-001/report.json','all25_screen_pending':True,'new_native_run_pending':True,'source_face_ancestry_counts':face_counts,'source_diagnostic_sha256':diag_hash,'local_roundtrip_separation_repair':roundtrip_projection_audit,'native_witness_boundary_repair':native_witness_repair,'passive_visual_only':True,'mechanics_changed':False,'myosim_route_added':False}
manifest['coverage']={'emitted_surface_count':1,'muscle_surface_count':1,'tendon_surface_count':0,'passive_append_added_stable_ids':[],'upstream_counts_not_recomputed_after_passive_append':['configured_surface_count','cancelled_opposite_face_pair_count','opposite_face_pair_cancellation_surface_count']}
manifest['runtime_binding']='Appended stable164 is a one-hot registered head visual reference surface; no active MyoSim route or jaw mechanics.'
manifest['evidence_boundary']='Inferred registered anatomy, exact bone/skin clearance and 25 captured-pose intersection evidence; not an individual-specific measurement or native anatomical admission.'
mp=subpath.with_suffix('.manifest.json'); mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
subset=pc._read_nhtiss4(subpath); sm=json.loads(mp.read_text()); pc._validate_passive_payload_metadata(subset,sm); pc._passive_append_rows(sm,(164,)); expected=pc._passive_surface_append_bytes(parent,subset,(164,))
loc=pc._row_slices(subset,subset['records'][0])
req(loc['local_faces']==faces.tolist() and loc['vertex_bytes']==bytes(vertex_blob) and loc['binding_bytes']==local159['binding_bytes'],'adapter row round-trip differs from candidate')
report={'schema':'offline.temporalis.passive-surface-subset-export.v1','status':'experimental_local_native_witness_passed_all25_screen_and_new_native_pending','subset_payload_path':str(subpath),'subset_payload_sha256':sha(subpath),'subset_manifest_path':str(mp),'subset_manifest_sha256':sha(mp),'candidate_npz_path':str(C),'candidate_npz_sha256':sha(C),'local_native_witness_report_path':str(RP),'local_native_witness_report_sha256':sha(RP),'candidate040_baseline_report_path':str(ROUNDTRIP),'candidate040_baseline_report_sha256':sha(ROUNDTRIP),'all25_screen_pending_path':'/Users/n/numi-human-exact-normalization-m6-1348/reports/temporalis-clearance-002/all25-candidate-screen-164-043/run-002/result-001/report.json','append_parent_payload_path':str(PARENT),'append_parent_payload_sha256':sha(PARENT),'append_parent_manifest_path':str(PM),'append_parent_manifest_sha256':sha(PM),'adapter_path':str(ADAPTER),'adapter_sha256':sha(ADAPTER),'payload_counts':{'surface_count':1,'binding_count':1,'vertex_count':nv,'index_count':ni,'stable_id':164,'face_ancestry_counts':face_counts},'source_identity_matches_parent':{'registration_fingerprint32':f'{fp:08x}','myosim_source_archive_sha256':sd.hex(),'binding_row159_sha256':bind_sha},'adapter_append_preflight_passed':True,'experimental_only':True,'all25_screen_pending':True,'new_native_run_pending':True,'parent_plus_subset_composition_bytes_sha256':hashlib.sha256(expected).hexdigest(),'checks':{'candidate_arrays_exactly_match_nhtiss_row':True,'local_eight_capture_native_witness_report_and_pins_verified':True,'all25_screen_pending':True,'new_native_run_pending':True,'source_lift_bone_cut_skin_patch_ancestry_retained':True,'roundtrip_correction_vector_and_both_normal_projections_pinned':True,'native_witness_boundary_adjustment_pinned':True,'passive_head_support_only':True}}
(OUT/'subset-export-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
