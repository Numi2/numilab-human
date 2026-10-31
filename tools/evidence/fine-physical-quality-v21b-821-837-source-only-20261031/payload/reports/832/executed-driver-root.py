from pathlib import Path
import json,hashlib,subprocess,importlib.util,sys,numpy as np
h=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");r=h/'all-quality-831-root';out=h/'all-quality-832-root';out.mkdir(exist_ok=False)
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
b=h/'typed-I-immediate-parent-alias-rebind-002/candidate-I-typed-current.npz'
c=h/'typed-I-immediate-parent-alias-rebind-003/run-831/candidate-I-typed-current.npz'
assert pin(b)['sha256']=='bde83424baeac5aa8f36067cceb774b89d67770a917c08074475d7eca89d9c5c'
assert pin(c)['sha256']=='7226d29b7777914cf5ebc6bc7be1e07703ebe3d867ff08ee36f7bf2382d2fc03'
cfg=json.loads((r/'proposal/775-context-config.json').read_text())
cfg['baseline_candidates']['I']=pin(b);cfg['candidate_surfaces']['I']=pin(c)
cfg['canonical_typed_parent_equivalence_receipts']=[pin(h/'typed-I-immediate-parent-alias-rebind-002/receipt.json'),pin(h/'typed-I-immediate-parent-alias-rebind-003/run-831/receipt.json')]
cp=out/'context-config.json';cp.write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
logs=out/'proposal-audit-logs';logs.mkdir()
cmd=["/Users/numi/numi-human-prep-venv-20261005/bin/python",str(h/'combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_805_root.py'),'--config',str(cp),'--out',str(out/'proposal-exact-805')]
with (logs/'audit805.stdout.log').open('w') as o,(logs/'audit805.stderr.log').open('w') as e:z=subprocess.run(cmd,stdout=o,stderr=e)
(logs/'audit805.command.json').write_text(json.dumps({'command':cmd,'returncode':z.returncode},indent=2)+'\n')
assert z.returncode==0,(logs/'audit805.stderr.log').read_text()[-2500:]
ap=out/'proposal-exact-805/report.json';audit=json.loads(ap.read_text());assert audit['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
parent=h/'all-quality-827-root/quality-working-state.json';state=json.loads(parent.read_text());raw=json.loads((r/'proposal/state.json').read_text())
rp=r/'proposal/independent-replay.json';replay=json.loads(rp.read_text());assert replay['status']=='PASS_INDEPENDENT_SOURCE_REPLAY';assert replay['run_state']==pin(r/'proposal/state.json')
state['source_state']=pin(parent);state['source_surfaces']['I']=pin(c);state['exact_context']=pin(ap)
for role in ['Skin','Fat']:
 v=raw['outputs'][role.lower()+'-coupled-anchor-volume-proposal.npz'];qc=replay['quality'][role]
 state[role+'_current']={**v,'native_quality_failures':qc['native_bad_after'],'nonpositive':0,'cells':qc['cells'],'boundary_source_metadata':'PENDING_CANONICAL_REGENERATION_FROM_IMMEDIATE_PARENT_UNIONS'}
state.update({'surface_crossings':0,'physical_admission':False,'native_steps':0,'mass_mapping':'PENDING','coupled_anchor_geometry_replay':pin(rp),'source_parent_alias_rebind':cfg['canonical_typed_parent_equivalence_receipts'],'construction_status':'QUALITY_ONLY_WORKING_STATE_NOT_PHYSICALLY_ADMITTED','typed_217_admission':'PENDING_EXPLICIT_ADAPTER','boundary_source_binding_status':'PENDING_CANONICAL_REBIND'})
dest=out/'quality-working-state.json';dest.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
s=importlib.util.spec_from_file_location('root832source',h/'verify_current_source_evidence_807_root.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
proof=m.verify_current_source_evidence(dest);(out/'source-proof.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS_CURRENT_SOURCE_AND_CELL_QUALITY_WORKING_STATE','state':pin(dest),'source_proof':pin(out/'source-proof.json'),'Skin':45,'Fat':183,'physical_admission':False,'native_steps':0,'metadata':'PENDING_CANONICAL_REGENERATION'}),flush=True)
