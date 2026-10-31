from pathlib import Path
import hashlib, json
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
P=H/'d-shortedge-quotient-414-next-001-input-attempt004'
P.mkdir(parents=True,exist_ok=False)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:np.array(z[k],copy=True) for k in z.files}
base=H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/fat-D411-379-plus-comp1-run041-composed.npz'; base_sha='14f06beaf20a2f78396585e7868df831cff4b9f27e3401e271d7643175073a77'
comp1_run=H/'fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24/connected-fixed-surface-volume-proposal.npz'; comp1_run_sha='e11249bd297a7e08ae8d74fe406439289dfaf2f1f3069d1ebf83876e3d170a15'
comp1_ref=H/'fat614-yy-volume-replay-004/run-035-current-407-target-rebind/rebound/fat-current-typed-target-rebound.npz'; comp1_ref_sha='38644a69f8b07409df9df09cb79cafc216edc5c223c395a040e4f2c017d8f237'
comp1_shell=H/'fat614-yy-volume-replay-004/run-037-D407-target-cavity-prep/connected-cavity-shells.npz'; comp1_shell_sha='d0b813675dbbf5fb7a8ef41b0b961b67f104de769d09afab3ae3e5c7aeb7e089'
comp0_shell=H/'fat-comp0-connected-warning-repair-412/run-014-current-comp0-edge-star-cavity/current-cavity-shells.npz'; comp0_sha='ec5de4831f13f23b0cf0f758f87b9f301652faeda893a5af9ff4b8707a653e63'
comp0_report=H/'fat-comp0-connected-warning-repair-412/run-014-current-comp0-edge-star-cavity/report.json'; comp0_report_sha='43a76d7c05b5884912d48a1f22601345fbf51585c49c51ba8d8359c3aabb4903'
comp0_mask=H/'fat-comp0-connected-warning-repair-412/run-014-current-comp0-edge-star-cavity/comp0-full-edge-star-exclusion.npz'; comp0_mask_sha='bfb88500119d48ff09c3af5358a405302ea29f152252d7b8dccbd6b4275ac893'
for p,w in [(base,base_sha),(comp1_run,comp1_run_sha),(comp1_ref,comp1_ref_sha),(comp1_shell,comp1_shell_sha),(comp0_shell,comp0_sha),(comp0_report,comp0_report_sha),(comp0_mask,comp0_mask_sha)]:
    got=sha(p)
    if got!=w: raise ValueError(f'input pin mismatch {p}: {got} != {w}')
V=load(base); VT=np.asarray(V['tetrahedra'],np.int64); VP=np.asarray(V['points_world_m'],np.float32)
cell_rows=np.asarray(V['source_comp1_output_cell_rows'],np.int64); added=np.asarray(V['source_comp1_added_point_rows'],np.int64)
if len(cell_rows)!=2212 or len(added)!=3: raise ValueError('unexpected recorded comp1 replacement map')
if np.any(cell_rows<0) or np.any(cell_rows>=len(VT)): raise ValueError('comp1 cell row outside 414')
cell_points=np.unique(VT[cell_rows].reshape(-1)).astype(np.int64)
S=load(comp1_shell); R=load(comp1_ref); oldp=np.asarray(R['points_world_m'],np.float32)
shell_rows=np.asarray(S['global_point_rows'],np.int64)
if len(shell_rows)!=1036: raise ValueError('run037 comp1 shell point roster changed')
lookup={np.ascontiguousarray(x).tobytes():i for i,x in enumerate(VP)}
if len(lookup)!=len(VP): raise ValueError('414 volume point coordinates are not unique')
mapped=np.asarray([lookup.get(np.ascontiguousarray(oldp[int(r)]).tobytes(),-1) for r in shell_rows],np.int64)
if np.any(mapped<0) or len(np.unique(mapped))!=len(mapped): raise ValueError('run037 comp1 shell points fail exact one-to-one current 414 crosswalk')
points=np.unique(np.concatenate((cell_points,added,mapped))).astype(np.int64)
if np.any(points<0) or np.any(points>=len(VP)): raise ValueError('comp1 current protected point row out of range')
adapter=P/'comp1-current-output-protection.npz'
np.savez_compressed(adapter,input_cell_rows=cell_rows,global_point_rows=points,run037_shell_point_rows=shell_rows,run037_shell_point_rows_to_current_414=mapped)
adapter_sha=sha(adapter)
comp1_doc={
 'schema':'h706.d-shortedge-current-comp1-output-protection.v1','status':'PASS_EXACT_414_OUTPUT_CELL_AND_FULL_1036_POINT_CROSSWALK',
 'source_volume_path':str(base),'source_volume_sha256':base_sha,
 'source_comp1_run041_path':str(comp1_run),'source_comp1_run041_sha256':comp1_run_sha,
 'source_comp1_run037_shell_path':str(comp1_shell),'source_comp1_run037_shell_sha256':comp1_shell_sha,
 'source_comp1_run037_reference_volume_path':str(comp1_ref),'source_comp1_run037_reference_volume_sha256':comp1_ref_sha,
 'current_output_cell_field':'source_comp1_output_cell_rows','current_added_point_field':'source_comp1_added_point_rows',
 'current_output_cell_count':int(len(cell_rows)),'current_patch_cell_point_count':int(len(cell_points)),
 'run037_shell_point_count':int(len(shell_rows)),'run037_shell_point_mapped_exactly':int(len(mapped)),
 'run037_shell_point_map_rows_sha256':hashlib.sha256(np.ascontiguousarray(mapped).tobytes()).hexdigest(),
 'protected_current_point_count':int(len(points)),
 'output_cell_rows_sha256':hashlib.sha256(np.ascontiguousarray(cell_rows).tobytes()).hexdigest(),
 'protected_current_point_rows_sha256':hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
 'adapter_path':str(adapter),'adapter_sha256':adapter_sha,
 'interpretation':'Protection-only adapter. It uses the exact 414 output rows for the accepted comp1 replacement and all patch vertices, plus the complete 1,036-point old shell mapped one-to-one by exact F32 coordinate to current 414. No geometry is changed.'}
comp1_report=P/'comp1-current-output-protection.json'; comp1_report.write_text(json.dumps(comp1_doc,indent=2,sort_keys=True)+'\n')
D_typed=H/'fat-comp0-connected-warning-repair-412/run-007-final-D411-comp1-flips/candidate-D-three-warning-flips-typed.npz'; D_typed_sha='5bcbe4c42618b84399edd9e878c192598efaa0c7c213b84b22ea85be1e40fd40'
D=H/'fat-comp0-connected-warning-repair-412/run-009-final-D411-comp1-flips-exact-adapter/candidate-D-for-exact-appended-audit.npz'; D_sha='707bdf3ac95fd0dc9539c148b9bad8e31207dab515d75cd21ebe039a38cb2703'
D_adapter_receipt=H/'fat-comp0-connected-warning-repair-412/run-009-final-D411-comp1-flips-exact-adapter/adapter-receipt.json'; D_adapter_receipt_sha='5de5ed283b09c1ad5b184b47241ba010d817986529e2c8920e7e06a005a694b2'
I=H/'paired-I-interface-399-root/iteration002/typed365/candidate-I-typed-current.npz'; I_sha='fec15d6b59e9094952845cbeaaf418b91a4b2f5d0405d1c58b7e3e2e752ec406'
O=H/'outer-O-witness-driven-sequence-448-root/iteration002/typed434001/candidate-O-typed-current.npz'; O_sha='7a3954dc5b655e6ac0601645a1fcaa809710b89b142bed63e13d477c89c66dc5'
Fat=H/'fat-comp0-connected-warning-repair-412/run-012-final-D-current-target-rebind/fat-current-typed-target-rebound.npz'; Fat_sha='72c851692580a639f89a4916bc4bf5c38331be9b6def89322eccf02ee66ea6ec'
context=H/'fat-comp0-connected-warning-repair-412/run-009-final-D411-comp1-flips-exact-adapter/exact-audit/report.json'; context_sha='44962483a961203a14c52b03eb954e87b2d4827e066f38567074e3d4b75c1a28'
source336=H/'fat-inferred-identity-mask-336-root/fat-run001/fixed-boundary-volume-proposal.npz'; source336_sha='2fc5aa8930fe50f940b071a954c0393b8e41afee514d64ace81e1c419e2b1905'
witness=H/'native-boundary-edge-witness-census-350-root/fat-short-surface-edge-witnesses.npz'; witness_sha='f5242f1ab6d9f1a74f5d2fd043c21c025dee1ac96d640fc91e4fe60f4063902d'
routes=H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json'; routes_sha='ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71'
back=H/'fixed-boundary-quality-topology-196/collapse-batch-004-from-003-001/back412-scope-audit-001/report.json'; back_sha='d20f1fd175b699a00dbe7077aae33b787054126a7d6bfef28c6d25894b30db0e'
preflight_owner=H/'prepare_d_short_edge_preflight_385_run005.py'; preflight_owner_sha='3cbae50bb91fb02021b3489dc679216ffd8eb7f0548bdcbf98fb77edbb0d1517'
transfer_owner=H/'transfer_surface_contractions_indexed_379_root.py'; transfer_owner_sha='ea9135fe6c19efc644bd8429bd01e5f36547f304e86da9214aa1bb594441da37'
preflight_414_source=Path('/tmp/prepare_d_short_edge_preflight_414.py')
preflight_414= P/'prepare_d_short_edge_preflight_414.py'
preflight_414.write_bytes(preflight_414_source.read_bytes())
preflight_414_sha=sha(preflight_414)
for p,w in [(D_typed,D_typed_sha),(D,D_sha),(D_adapter_receipt,D_adapter_receipt_sha),(I,I_sha),(O,O_sha),(Fat,Fat_sha),(context,context_sha),(source336,source336_sha),(witness,witness_sha),(routes,routes_sha),(back,back_sha),(preflight_owner,preflight_owner_sha),(transfer_owner,transfer_owner_sha)]:
    got=sha(p)
    if got!=w: raise ValueError(f'input pin mismatch {p}: {got} != {w}')
# The exact-context report used the append-only D adapter; prove its geometry is bit-identical to the final typed D source before using it for this preflight.
A=load(D_typed); B=load(D)
for key in ('points_world_m_f32','points_world_m','faces'):
    if not np.array_equal(A[key],B[key]): raise ValueError(f'D context adapter changes geometry array {key}')
adapter_receipt_doc=json.loads(D_adapter_receipt.read_text())
if adapter_receipt_doc.get('candidate_sha256')!=D_typed_sha or adapter_receipt_doc.get('adapter_sha256')!=D_sha:
    raise ValueError('D context adapter receipt does not bind the final typed D source')
geometry_adapter_doc={'schema':'h706.d-shortedge-final-D-context-geometry-adapter.v1','status':'PASS_BIT_IDENTICAL_FINAL_TYPED_D_GEOMETRY','typed_D_path':str(D_typed),'typed_D_sha256':D_typed_sha,'context_adapter_path':str(D),'context_adapter_sha256':D_sha,'adapter_receipt_path':str(D_adapter_receipt),'adapter_receipt_sha256':D_adapter_receipt_sha,'points_f32_bit_identical':True,'points_f64_bit_identical':True,'faces_identical':True,'point_count':int(len(A['points_world_m_f32'])),'face_count':int(len(A['faces']))}
geometry_adapter_path=P/'D-context-geometry-adapter.json'; geometry_adapter_path.write_text(json.dumps(geometry_adapter_doc,indent=2,sort_keys=True)+'\n')
def pin(p,h): return {'path':str(p),'sha256':h}
cfg={'schema':'h706.d-short-edge-current-preflight-input.v2','artifact_root':str(H),'batch_id':'D-shortedge-414-D411-comp1-flips-current-10um-batch-001','d_witness_layer':2,'output_directory':str(H/'d-shortedge-quotient-414-next-001-preflight-attempt004'),'parameters':{'max_edge_length_m':1.0e-5,'maximum_candidates':8192,'max_accepted':8192,'max_trials':100000,'wall_budget_seconds':90.0},'source_preflight_385':str(preflight_owner),'source_preflight_385_sha256':preflight_owner_sha,'source_preflight_414':{'path':str(preflight_414),'sha256':preflight_414_sha},'transfer_owner_379':str(transfer_owner),'transfer_owner_379_sha256':transfer_owner_sha,'inputs':{'O':pin(O,O_sha),'I':pin(I,I_sha),'D':pin(D,D_sha),'fat_volume':pin(Fat,Fat_sha),'witness_census':pin(witness,witness_sha),'source_Fat336':pin(source336,source336_sha),'cavity_shells':pin(comp0_shell,comp0_sha),'cavity_reference_volume':pin(base,base_sha),'comp1_cavity_shells':pin(adapter,adapter_sha),'comp1_cavity_reference_volume':pin(base,base_sha),'protected_routes':pin(routes,routes_sha),'complete_context_audit':pin(context,context_sha),'back_scope_report':pin(back,back_sha)}}
config=P/'run-input.json'; config.write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
source_script=P/'prepare_d414_preflight_003.py'; source_script.write_bytes(Path(__file__).read_bytes()); script_hash=sha(source_script)
manifest={'schema':'h706.d-shortedge-414-next-batch-preparation-manifest.v1','status':'PASS_INPUTS_AND_FULL_PROTECTION_PREPARED_PREFLIGHT_NOT_RUN','preparation_script_path':str(source_script),'preparation_script_sha256':script_hash,'preflight_414_script_path':str(preflight_414),'preflight_414_script_sha256':preflight_414_sha,'config_path':str(config),'config_sha256':sha(config),'comp1_protection_path':str(adapter),'comp1_protection_sha256':adapter_sha,'comp1_protection_report_path':str(comp1_report),'comp1_protection_report_sha256':sha(comp1_report),'comp0_shell_path':str(comp0_shell),'comp0_shell_sha256':comp0_sha,'comp0_report_path':str(comp0_report),'comp0_report_sha256':comp0_report_sha,'comp0_edge_star_mask_path':str(comp0_mask),'comp0_edge_star_mask_sha256':comp0_mask_sha,'current_volume_path':str(base),'current_volume_sha256':base_sha,'current_target_rebind_path':str(Fat),'current_target_rebind_sha256':Fat_sha,'current_typed_D_path':str(D_typed),'current_typed_D_sha256':D_typed_sha,'current_D_context_adapter_path':str(D),'current_D_context_adapter_sha256':D_sha,'D_context_geometry_adapter_receipt_path':str(geometry_adapter_path),'D_context_geometry_adapter_receipt_sha256':sha(geometry_adapter_path),'current_typed_I_path':str(I),'current_typed_I_sha256':I_sha,'current_typed_O_path':str(O),'current_typed_O_sha256':O_sha,'complete_context_audit_path':str(context),'complete_context_audit_sha256':context_sha,'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN'}
manifest_path=P/'preparation-manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':manifest['status'],'config':str(config),'config_sha256':sha(config),'comp1_protection_sha256':adapter_sha,'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'comp1_cells':len(cell_rows),'comp1_cell_points':len(cell_points),'comp1_full_shell_points':len(mapped),'comp1_protected_points':len(points)},sort_keys=True))
