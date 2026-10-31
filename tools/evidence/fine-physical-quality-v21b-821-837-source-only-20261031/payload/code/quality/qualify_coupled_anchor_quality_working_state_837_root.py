#!/usr/bin/env python3
"""Qualify current-source and cell geometry for continued tissue construction; no physical admission."""
from pathlib import Path
import argparse,hashlib,json,subprocess,importlib.util,sys
H=Path(__file__).resolve().parent
PY=sys.executable
def pin(p):
 p=Path(p).resolve()
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 r=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 rawp=r/'proposal/state.json';raw=json.loads(rawp.read_text());parent=json.loads(a.parent.read_text());cfgp=r/'inputs/run-config.json';cfg=json.loads(cfgp.read_text())
 assert cfg['baseline_surfaces']==parent['source_surfaces']
 for role in ['Skin','Fat']:
  assert cfg['volumes'][role]['path']==parent[role+'_current']['path'] and cfg['volumes'][role]['sha256']==parent[role+'_current']['sha256']
 replay_owner=H/'coupled-anchor-collapse-780-root/coupled_anchor_collapse_replay_789_root.py'
 assert pin(replay_owner)['sha256']=='5fcefd6745b09754e6414bfd1d86a364b76fcfd8946fb077d8636ac5c1663052'
 rp=out/'independent-replay.json'
 cmd=[PY,str(replay_owner),'--mode','replay','--config',str(cfgp),'--run-dir',str(r/'proposal'),'--out',str(rp)]
 with (out/'replay.log').open('w') as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if z.returncode:raise RuntimeError((out/'replay.log').read_text()[-3500:])
 rep=json.loads(rp.read_text());assert rep['status']=='PASS_INDEPENDENT_SOURCE_REPLAY' and rep['run_state']==pin(rawp)
 cp=r/'proposal/775-context-config.json';context=json.loads(cp.read_text())
 assert context['baseline_candidates']==parent['source_surfaces']
 assert context['candidate_surfaces']['I']==raw['outputs']['candidate-I-typed-current.npz']
 assert context['candidate_surfaces']['O']==parent['source_surfaces']['O'] and context['candidate_surfaces']['D']==parent['source_surfaces']['D']
 logs=out/'proposal-audit-logs';logs.mkdir()
 auditor=H/'combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_805_root.py'
 assert pin(auditor)['sha256']=='0f41e4114eeb56032d4bb6a6c39484d2e8dd0de52ff977c7211a800273101234'
 cmd=[PY,str(auditor),'--config',str(cp),'--out',str(out/'proposal-exact-805')]
 with (logs/'audit805.stdout.log').open('w') as o,(logs/'audit805.stderr.log').open('w') as e:z=subprocess.run(cmd,stdout=o,stderr=e)
 (logs/'audit805.command.json').write_text(json.dumps({'command':cmd,'returncode':z.returncode},indent=2)+'\n')
 if z.returncode:raise RuntimeError((logs/'audit805.stderr.log').read_text()[-3500:])
 auditp=out/'proposal-exact-805/report.json';audit=json.loads(auditp.read_text())
 if audit['status']!='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185':
  print(json.dumps({'status':'REJECT_EXACT_SURFACE','report':pin(auditp),'quality_working_state':None}));return
 state=parent.copy();state['source_state']=pin(a.parent);state['source_surfaces']=context['candidate_surfaces'];state['exact_context']=pin(auditp)
 for role in ['Skin','Fat']:
  v=raw['outputs'][role.lower()+'-coupled-anchor-volume-proposal.npz'];qc=rep['quality'][role]
  state[role+'_current']={**v,'native_quality_failures':qc['native_bad_after'],'nonpositive':0,'cells':qc['cells'],'boundary_source_metadata':'PENDING_CANONICAL_REGENERATION_FROM_IMMEDIATE_PARENT_UNIONS'}
 state.update({'surface_crossings':0,'physical_admission':False,'native_steps':0,'mass_mapping':'PENDING','coupled_anchor_geometry_replay':pin(rp),'construction_status':'QUALITY_ONLY_WORKING_STATE_NOT_PHYSICALLY_ADMITTED','typed_217_admission':'PENDING_EXPLICIT_ADAPTER','boundary_source_binding_status':'PENDING_CANONICAL_REBIND','quality_working_state_qualifier':pin(__file__)})
 dest=out/'quality-working-state.json'
 dest.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
 verifier=H/'verify_current_source_evidence_807_root.py';assert pin(verifier)['sha256']=='9a539227c072936a95b3a9457405521998fb40e5df6139f08d7877fbb49c3f62'
 proof=module(verifier,'qualifier837source').verify_current_source_evidence(dest);pp=out/'source-proof.json';pp.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS_CURRENT_SOURCE_AND_CELL_QUALITY_WORKING_STATE','state':pin(dest),'source_proof':pin(pp),'Skin':state['Skin_current']['native_quality_failures'],'Fat':state['Fat_current']['native_quality_failures'],'physical_admission':False,'native_steps':0,'metadata':'PENDING_CANONICAL_REGENERATION'}))
if __name__=='__main__':main()
