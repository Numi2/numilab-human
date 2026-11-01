#!/usr/bin/env python3
"""Prepare a byte-pinned archive plan for the source-only 943 geometry snapshot."""
import datetime, hashlib, json
from pathlib import Path
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706').resolve()
OUT=H/'quality-943-source-snapshot-20261101-plan'
R943=H/'all-quality-943-expanded-joint-root'
R937=H/'all-quality-937-b5-backtracking-root'
R939=H/'all-quality-939-b5-backtracking-root'
SOURCES=[
 ('current_943_proposal_state','all-quality-943-expanded-joint-root/proposal/state.json','metadata'),
 ('current_943_verified_state','all-quality-943-expanded-joint-root/qualified/verified-state.json','metadata'),
 ('current_943_qualification_receipt','all-quality-943-expanded-joint-root/qualified/receipt.json','proof'),
 ('current_943_independent_v26c_replay','all-quality-943-expanded-joint-root/qualified/independent-v26c-replay.json','proof'),
 ('current_943_base_volume_replay','all-quality-943-expanded-joint-root/qualified/independent-v26c-replay-base-volume-replay.json','proof'),
 ('current_943_legacy_compat_state','all-quality-943-expanded-joint-root/qualified/independent-v26c-replay-legacy-volume-compat-state.json','metadata'),
 ('current_806_proof','all-quality-943-expanded-joint-root/qualified/proposal-exact-806/report.json','proof'),
 ('current_context_config','all-quality-943-expanded-joint-root/proposal/context-config.json','configuration'),
 ('current_witness_face_remap','all-quality-943-expanded-joint-root/proposal/witness-face-remap.json','provenance'),
 ('current_lp_diagnostics','all-quality-943-expanded-joint-root/proposal/lp-diagnostics.json','diagnostics'),
 ('current_fat_writer_receipt','all-quality-943-expanded-joint-root/proposal/fat-volume-cow-writer-receipt.json','provenance'),
 ('current_i_writer_receipt','all-quality-943-expanded-joint-root/proposal/i-source-cow-writer-receipt.json','provenance'),
 ('current_skin_writer_receipt','all-quality-943-expanded-joint-root/proposal/skin-volume-cow-writer-receipt.json','provenance'),
 ('current_actual_back_fat_protected_rows','all-quality-943-expanded-joint-root/proposal/actual-back-fat-protected-rows.npz','source_geometry'),
 ('current_O_typed_source','all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz','source_geometry'),
 ('current_I_typed_source','all-quality-943-expanded-joint-root/proposal/candidate-I-typed-current.npz','source_geometry'),
 ('current_D_typed_source','all-quality-939-b5-backtracking-root/proposal/candidate-D-typed-current.npz','source_geometry'),
 ('current_Fat_volume_proposal','all-quality-943-expanded-joint-root/proposal/fat-coupled-quality-point-proposal.npz','volume_proposal'),
 ('current_Skin_volume_proposal','all-quality-943-expanded-joint-root/proposal/skin-coupled-quality-point-proposal.npz','volume_proposal'),
 ('O_parent_937_proposal_state','all-quality-937-b5-backtracking-root/proposal/state.json','lineage_metadata'),
 ('O_parent_937_verified_state','all-quality-937-b5-backtracking-root/qualified/verified-state.json','lineage_metadata'),
 ('O_parent_937_receipt','all-quality-937-b5-backtracking-root/qualified/receipt.json','lineage_proof'),
 ('D_parent_939_proposal_state','all-quality-939-b5-backtracking-root/proposal/state.json','lineage_metadata'),
 ('D_parent_939_verified_state','all-quality-939-b5-backtracking-root/qualified/verified-state.json','lineage_metadata'),
 ('D_parent_939_receipt','all-quality-939-b5-backtracking-root/qualified/receipt.json','lineage_proof'),
 ('current_builder_command','all-quality-943-expanded-joint-root/builder.command.json','execution_provenance'),
 ('current_checkpoint_command','all-quality-943-expanded-joint-root/qualified/checkpoint-logs/checkpoint.command.json','execution_provenance'),
 ('current_806_command','all-quality-943-expanded-joint-root/qualified/proposal-audit-logs/audit806.command.json','execution_provenance'),
 ('current_verifier_command','all-quality-943-expanded-joint-root/qualified/verification-logs/verifier.command.json','execution_provenance'),
]
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def walk_pin(node,target,digest):
 if isinstance(node,dict):
  pv=node.get('path'); sv=node.get('sha256')
  if isinstance(pv,str) and Path(pv).is_absolute() and str(Path(pv).resolve())==target and sv==digest:return True
  return any(walk_pin(v,target,digest) for v in node.values())
 if isinstance(node,list):return any(walk_pin(v,target,digest) for v in node)
 return False
state=json.loads((R943/'proposal/state.json').read_text())
verified=json.loads((R943/'qualified/verified-state.json').read_text())
receipt=json.loads((R943/'qualified/receipt.json').read_text())
replay=json.loads((R943/'qualified/independent-v26c-replay.json').read_text())
base=json.loads((R943/'qualified/independent-v26c-replay-base-volume-replay.json').read_text())
assert state.get('status')=='SOURCE_ONLY_BATCH_PROPOSAL_PENDING_CURRENT_ALL185_AND_NATIVE_GATES'
assert verified.get('physical_admission') is False and verified.get('native_steps')==0
assert receipt.get('physical_admission') is False and receipt.get('native_steps')==0
assert replay.get('physical_admission') is False and replay.get('native_steps')==0
assert base.get('physical_admission') is False and base.get('native_steps')==0
assert base.get('mass_reconciliation',{}).get('status')=='PENDING_VOLUME_CHANGE_RECONCILIATION'
assert state['Fat_current']['native_quality_failures']==6
assert state['Skin_current']['native_quality_failures']==6
# Scan only the explicitly selected JSON documents to document actual cross-pins.
rows=[]; selected_json=[]
for role,rel,kind in SOURCES:
 p=(H/rel).resolve()
 if p.suffix.lower()=='.json':
  selected_json.append((role,p,json.loads(p.read_text())))
for role,rel,kind in SOURCES:
 p=(H/rel).resolve()
 if p.is_symlink() or not p.is_file() or p!=H/rel:
  raise RuntimeError(f'not a regular canonical file: {p}')
 digest=sha256(p); size=p.stat().st_size
 pin_docs=[]
 if p.suffix.lower()=='.npz':
  for drole,dp,doc in selected_json:
   if walk_pin(doc,str(p),digest):pin_docs.append(drole)
 rows.append({'role':role,'kind':kind,'path':str(p),'source_relative_path':p.relative_to(H).as_posix(),'size_bytes':size,'sha256':digest,'pin_status':'CROSS_PINNED_BY_SELECTED_JSON' if pin_docs else 'DIRECT_PLAN_SHA_ONLY','pinning_selected_json_roles':pin_docs})
# Persist explicit provenance/scope statement with the plan.
plan={
 'schema':'numi-human.source-only-geometry-snapshot-plan.v1',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),
 'preparation_status':'READY_FOR_REVIEW_LIVE_PAYLOAD_HASH_PENDING',
 'snapshot':{
  'id':'quality-943-best-source-only-snapshot-20261101-001',
  'qualification_root':'all-quality-943-expanded-joint-root',
  'status':'SOURCE_ONLY_BATCH_PROPOSAL_PENDING_CURRENT_ALL185_AND_NATIVE_GATES',
  'fat_native_quality_failures':6,'skin_native_quality_failures':6,
  'physical_admission':False,'native_steps':0,
  'mass_reconciliation_status':'PENDING_VOLUME_CHANGE_RECONCILIATION',
  'source_only_qualification':True,
  'original_ancestry_closure':'INCOMPLETE: archives current O/I/D, current Skin/Fat, 943 proof/config, and immediate O/D parent metadata; it does not include the complete historical source/volume ancestry.'},
 'source_root':str(H),
 'chunk':{'id':'quality-943-best-source-snapshot-20261101-001','ordinal':1,'file_count':len(rows),'original_bytes':sum(r['size_bytes'] for r in rows),'retirement_enabled':False},
 'source_workflow':{'repository':'Numi2/numilab-human','workflow_commit':'a7f784ecdd99f8b4ea70219304fff15b8d36cf85','archive_layout':'content-addressed blobs/<sha256> with archive-index.json','builder_reuse_path':str(OUT/'build_943_snapshot.py'),'independent_verifier_path':str(OUT/'verify_943_snapshot.py'),'full_original_ancestry_included':False},
 'files':rows,
 'selection':{'no_current_files_retired':True,'all_current_and_source_NPZs_remain_at_original_paths':True,'all_current_files_hashed_at_plan_creation':True,'NPZ_cross_pins_documented_but_no_original_ancestry_closure_claim':True},
 'release_target':{'repository':'Numi2/numilab-human','tag':'human-quality-source-snapshot-20261101-943-001','asset_name':'quality-943-source-snapshot-20261101-001.tar.zst','visibility_note':'Use authenticated repository release; do not change access policy.'},
}
path=OUT/'selection-plan.json';path.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':plan['preparation_status'],'file_count':len(rows),'original_bytes':plan['chunk']['original_bytes'],'plan_path':str(path),'plan_sha256':sha256(path),'npz_cross_pin_status':{r['role']:{'pin_status':r['pin_status'],'pinning_selected_json_roles':r['pinning_selected_json_roles']} for r in rows if r['path'].endswith('.npz')},'source_only_status':plan['snapshot']},sort_keys=True))
