from pathlib import Path
import json,hashlib,subprocess,importlib.util,sys
h=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");r=h/'all-quality-826-root';out=h/'all-quality-827-root';out.mkdir(exist_ok=False)
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
logs=r/'proposal-audit-logs';logs.mkdir(exist_ok=False)
cmd=["/Users/numi/numi-human-prep-venv-20261005/bin/python",str(h/'combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py'),'--config',str(r/'verified-context-775.json'),'--out',str(r/'exact-775-recorded')]
with (logs/'audit775.stdout.log').open('w') as o,(logs/'audit775.stderr.log').open('w') as e:
 z=subprocess.run(cmd,stdout=o,stderr=e)
(logs/'audit775.command.json').write_text(json.dumps({'command':cmd,'returncode':z.returncode},indent=2)+'\n')
assert z.returncode==0
parent=h/'all-quality-824-root/round001/flips/verified-state.json'
state=json.loads(parent.read_text());raw=json.loads((r/'proposal/state.json').read_text())
replayp=r/'proposal/independent-replay.json';replay=json.loads(replayp.read_text())
assert replay['status']=='PASS_INDEPENDENT_SOURCE_REPLAY'
assert replay['run_state']==pin(r/'proposal/state.json')
auditp=r/'exact-775-recorded/report.json';audit=json.loads(auditp.read_text())
assert audit['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
state['source_state']=pin(parent);state['source_surfaces']['I']=raw['outputs']['candidate-I-typed-current.npz']
state['exact_context']=pin(auditp)
for role in ['Skin','Fat']:
 v=raw['outputs'][role.lower()+'-coupled-anchor-volume-proposal.npz'];qc=replay['quality'][role]
 state[role+'_current']={**v,'native_quality_failures':qc['native_bad_after'],'nonpositive':0,'cells':qc['cells'],'boundary_source_metadata':'PENDING_CANONICAL_REGENERATION_FROM_IMMEDIATE_PARENT_UNIONS'}
state.update({'source_audit_status':'PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY','surface_crossings':0,'source_metadata_current_step_only':True,'physical_admission':False,'native_steps':0,
 'coupled_anchor_geometry_replay':pin(replayp),'construction_status':'QUALITY_ONLY_WORKING_STATE_NOT_PHYSICALLY_ADMITTED','boundary_source_binding_status':'PENDING_CANONICAL_REBIND','typed_217_admission':'PENDING_EXPLICIT_ADAPTER','mass_mapping':'PENDING'})
dest=out/'quality-working-state.json';dest.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
p=h/'verify_current_source_evidence_807_root.py'
s=importlib.util.spec_from_file_location('root827source',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
proof=m.verify_current_source_evidence(dest)
(out/'source-proof.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS_QUALITY_ONLY_SOURCE_AND_CELL_REPLAY','state':pin(dest),'source_proof':pin(out/'source-proof.json'),'Skin':51,'Fat':188,'physical_admission':False,'metadata':'PENDING_CANONICAL_REGENERATION'}))
