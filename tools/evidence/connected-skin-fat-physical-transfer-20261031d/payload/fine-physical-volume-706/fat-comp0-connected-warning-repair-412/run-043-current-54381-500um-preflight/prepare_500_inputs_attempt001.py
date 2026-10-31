from pathlib import Path
import numpy as np, json, hashlib
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
root=H/'fat-comp0-connected-warning-repair-412'
base=root/'run-042-current-54381-normalized-rebind/fat-current-typed-target-rebound-54381-lineage-corrected.npz'
base_sha='5aa02086aac4b20e879332cd9d5ba64da636dd8a520de54e7b96b00225c7fe42'
mask=root/'run-037-local-conditioning-492/run-038-kernel-fans-297/protection-006/comp0-comp1-protected-stars.npz'
mask_sha='aaa7b1d01d354b6ad18a885359738b84f267e31cb60a9794665f43f98980365d'
mask_receipt=mask.parent/'receipt.json'
run=root/'run-043-current-54381-500um-preflight'
shell_dir=run/'current-protected-shells'
if run.exists(): raise FileExistsError(run)
run.mkdir(parents=True); shell_dir.mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
for p,h in [(base,base_sha),(mask,mask_sha)]:
 if sha(p)!=h: raise ValueError(f'pin mismatch {p}: {sha(p)} != {h}')
with np.load(base,allow_pickle=False) as z: V={k:np.array(z[k],copy=True) for k in z.files}
with np.load(mask,allow_pickle=False) as z: M={k:np.array(z[k],copy=True) for k in z.files}
ncell=len(V['tetrahedra']); npoint=len(V['points_world_m_f32'])
for name,cellkey,pointkey in [('comp0','comp0_patch_output_cell_rows','comp0_patch_point_rows'),('comp1','comp1_current_full_star_cell_rows','comp1_current_protected_point_rows')]:
 rows=np.asarray(M[cellkey],np.int64); pts=np.asarray(M[pointkey],np.int64)
 if np.any(rows<0) or np.any(rows>=ncell) or np.any(pts<0) or np.any(pts>=npoint): raise ValueError(f'{name} protection row outside current candidate')
 path=shell_dir/f'{name}-current-fullstar-shell.npz'
 np.savez_compressed(path,input_cell_rows=rows,global_point_rows=pts)
with np.load(shell_dir/'comp0-current-fullstar-shell.npz',allow_pickle=False) as z:C0={k:np.array(z[k],copy=True) for k in z.files}
with np.load(shell_dir/'comp1-current-fullstar-shell.npz',allow_pickle=False) as z:C1={k:np.array(z[k],copy=True) for k in z.files}
audit=H/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/exact-audit-197-005/report.json'
audit_sha='083c88aaae74455a9c5084f4097cd942bb8031d743e2cb34cd10e269463b42f4'
correction=root/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/metadata-lineage-correction-002/metadata-lineage-correction-receipt.json'
correction_sha='146d3c38065b5254f5fb2d9efbbaf35f92664bd3c3bf9ee77eb50fcb5b7e0625'
if sha(audit)!=audit_sha or sha(correction)!=correction_sha: raise ValueError('audit/correction pin mismatch')
bridge={
 'schema':'h706.inherited-exact-context-metadata-equivalence-bridge.v1',
 'status':'PASS_INHERITED_ALL185_GEOMETRY_IDENTICAL_METADATA_ONLY_BRIDGE',
 'scope':'all 185 source surfaces, both cavity walls, full O/I/D typed surfaces; inherited from audit-197; not a fresh intersection run',
 'source_audit':{'path':str(audit),'sha256':audit_sha,'status':json.loads(audit.read_text()).get('status'),'all185_scope':True},
 'role_candidates':{
  'O':{'path':str(H/'outer-O-connected-quality-506-root/candidate/typed434001/candidate-O-typed-current.npz'),'sha256':'0c3353060d3e675f7c907e8b0bbf4317b798b0c041cf9f25294b22bee9e38f0f'},
  'I':{'path':str(root/'run-031-threeD-oneI-safe-flips/prep-003/candidate-I-warning-flips-typed.npz'),'sha256':'81b285a4e12f11dfd4156bb2545bb86f5ccee900ef6021a11beff36954b00774'},
  'D':{'path':str(root/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/metadata-lineage-correction-002/candidate-D-comp0-480-boundary-split-typed-lineage-corrected-001.npz'),'sha256':'6451dfe347d42b247ecf9ca9c6baf14dd8f0c0bf7d8aa5bfca81c7fa8b6991e8','audited_geometric_parent_sha256':'3ef9353262b4821318c76a6e75f47a22101f62368d6546c3ea7026275bde5e7f','metadata_correction_receipt':{'path':str(correction),'sha256':correction_sha}}
 },
 'equivalence_basis':{'candidate_arrays_byte_identical_to_audited_D_split':True,'correction_receipt_sha256':correction_sha,'actual_697_loader_receipt_sha256':'6b39046f3648d571ceb4772f3cac3fd49c076bba1df4dfd4236129bd73b361e4','all185_intersection_claim_inherited_only':True},
 'admission':{'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN'}
}
bridge_path=run/'inherited-context-bridge-197-metadata-correction.json'
bridge_path.write_text(json.dumps(bridge,sort_keys=True,indent=2)+'\n')
old_cfg=json.loads((H/'d-shortedge-quotient-475-next-001-preflight-100um/run-input-475.json').read_text())
warn=old_cfg.get('protected_warning_point_sets',[])
O=H/'outer-O-connected-quality-506-root/candidate/typed434001/candidate-O-typed-current.npz'
I=root/'run-031-threeD-oneI-safe-flips/prep-003/candidate-I-warning-flips-typed.npz'
D=root/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/metadata-lineage-correction-002/candidate-D-comp0-480-boundary-split-typed-lineage-corrected-001.npz'
D697=root/'d-shortedge-quotient-480-next-001-preflight-100um/volume-transfer-001/comp0-boundary-split-001/metadata-lineage-correction-002/actual-697-003/actual-697-D-comp0-split-load.json'
source385=H/'prepare_d_short_edge_preflight_385_run005.py'
transfer379=H/'transfer_surface_contractions_indexed_379_root.py'
back=H/'fixed-boundary-quality-topology-196/collapse-batch-004-from-003-001/back412-scope-audit-001/report.json'
routes=H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json'
witness=H/'native-boundary-edge-witness-census-350-root/fat-short-surface-edge-witnesses.npz'
source336=H/'fat-inferred-identity-mask-336-root/fat-run001/fixed-boundary-volume-proposal.npz'
for p in [O,I,D,D697,source385,transfer379,back,routes,witness,source336]:
 if not p.is_file(): raise FileNotFoundError(p)
source_script=H/'d-shortedge-quotient-475-next-001-preflight-100um/prepare_d_short_edge_preflight_475.py'
script=run/'prepare_d_short_edge_preflight_500.py'
s=source_script.read_text()
s=s.replace('Config-driven, bounded D short-edge preflight derived from 385/379 owners.','Config-driven 500 um D short-edge preflight derived from the existing 475/385/379 owners.')
s=s.replace('max_edge <= 1e-4','max_edge <= 5e-4')
s=s.replace('positive and <= 100 um','positive and <= 500 um')
s=s.replace('older than the current 475 quotient topology','current full-star protection prepared in this volume row domain')
s=s.replace('"475_preflight"','"500_preflight"')
script.write_text(s)
config={
 'schema':'h706.d-short-edge-current-preflight-input.v2',
 'batch_id':'D-shortedge-current-500um-54381-protected-fullstars',
 'artifact_root':str(H),
 'config_owner':{'path':str(script),'sha256':sha(script),'parent_path':str(source_script),'parent_sha256':sha(source_script),'declared_change':'max edge construction cap only: 100 um to 500 um; all physics/topology/native gates unchanged'},
 'd_witness_layer':int(old_cfg.get('d_witness_layer',2)),
 'inputs':{
  'O':{'path':str(O),'sha256':'0c3353060d3e675f7c907e8b0bbf4317b798b0c041cf9f25294b22bee9e38f0f'},
  'I':{'path':str(I),'sha256':'81b285a4e12f11dfd4156bb2545bb86f5ccee900ef6021a11beff36954b00774'},
  'D':{'path':str(D),'sha256':'6451dfe347d42b247ecf9ca9c6baf14dd8f0c0bf7d8aa5bfca81c7fa8b6991e8'},
  'fat_volume':{'path':str(base),'sha256':base_sha},
  'witness_census':{'path':str(witness),'sha256':'f5242f1ab6d9f1a74f5d2fd043c21c025dee1ac96d640fc91e4fe60f4063902d'},
  'source_Fat336':{'path':str(source336),'sha256':'2fc5aa8930fe50f940b071a954c0393b8e41afee514d64ace81e1c419e2b1905'},
  'cavity_shells':{'path':str(shell_dir/'comp0-current-fullstar-shell.npz'),'sha256':sha(shell_dir/'comp0-current-fullstar-shell.npz')},
  'cavity_reference_volume':{'path':str(base),'sha256':base_sha},
  'comp1_cavity_shells':{'path':str(shell_dir/'comp1-current-fullstar-shell.npz'),'sha256':sha(shell_dir/'comp1-current-fullstar-shell.npz')},
  'comp1_cavity_reference_volume':{'path':str(base),'sha256':base_sha},
  'protected_routes':{'path':str(routes),'sha256':'ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71'},
  'complete_context_audit':{'path':str(bridge_path),'sha256':sha(bridge_path)},
  'back_scope_report':{'path':str(back),'sha256':'d20f1fd175b699a00dbe7077aae33b787054126a7d6bfef28c6d25894b30db0e'}
 },
 'source_preflight_385':str(source385),'source_preflight_385_sha256':'3cbae50bb91fb02021b3489dc679216ffd8eb7f0548bdcbf98fb77edbb0d1517',
 'transfer_owner_379':str(transfer379),'transfer_owner_379_sha256':'ea9135fe6c19efc644bd8429bd01e5f36547f304e86da9214aa1bb594441da37',
 'output_directory':str(run/'preflight-output'),
 'parameters':{'max_edge_length_m':5e-4,'maximum_candidates':8192,'max_accepted':8192,'max_trials':100000,'wall_budget_seconds':90.0},
 'protected_warning_point_sets':warn,
 'protection_sidecar':{'path':str(mask),'sha256':mask_sha,'receipt_path':str(mask_receipt),'receipt_sha256':sha(mask_receipt),'union_cells':33765,'union_points':11338,'comp0_cells':26383,'comp1_cells':7382,'comp1_registered_points':1036,'comp1_corner_points':2925,'comp1_source_output_rows':2212}
}
config_path=run/'run-input-500.json'
config_path.write_text(json.dumps(config,sort_keys=True,indent=2)+'\n')
shell_receipt={'schema':'h706.current-protected-shells-from-protection006.v1','status':'PASS_CURRENT_FULLSTAR_SHELLS_READY','current_candidate_path':str(base),'current_candidate_sha256':base_sha,'protection_sidecar_path':str(mask),'protection_sidecar_sha256':mask_sha,'shells':{},'admission':{'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN'}}
for name,p,rows,pts in [('comp0',shell_dir/'comp0-current-fullstar-shell.npz',len(C0['input_cell_rows']),len(C0['global_point_rows'])),('comp1',shell_dir/'comp1-current-fullstar-shell.npz',len(C1['input_cell_rows']),len(C1['global_point_rows']))]:
 shell_receipt['shells'][name]={'path':str(p),'sha256':sha(p),'selected_current_cells':rows,'seed_fullstar_points':pts,'reference_volume':'same normalized 54381 candidate'}
(shell_dir/'receipt.json').write_text(json.dumps(shell_receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps({'run':str(run),'script':str(script),'script_sha256':sha(script),'config':str(config_path),'config_sha256':sha(config_path),'bridge':str(bridge_path),'bridge_sha256':sha(bridge_path),'shell_receipt':str(shell_dir/'receipt.json'),'comp0_shell_sha256':sha(shell_dir/'comp0-current-fullstar-shell.npz'),'comp1_shell_sha256':sha(shell_dir/'comp1-current-fullstar-shell.npz'),'comp0_cells':len(C0['input_cell_rows']),'comp0_points':len(C0['global_point_rows']),'comp1_cells':len(C1['input_cell_rows']),'comp1_points':len(C1['global_point_rows']),'warning_rows':sorted({int(row) for rec in warn for row in rec.get('volume_point_rows',[])}),'max_edge_m':5e-4,'maximum_candidates':8192,'wall_budget_seconds':90},indent=2))
