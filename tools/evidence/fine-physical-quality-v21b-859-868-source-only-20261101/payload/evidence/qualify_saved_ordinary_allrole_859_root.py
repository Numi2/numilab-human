#!/usr/bin/env python3
from pathlib import Path
import importlib.util,json,sys,subprocess,hashlib
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
owner=H/'coupled-anchor-collapse-780-root/ordinary-allrole-v25/run_ordinary_allrole_v25_v3.py'
assert pin(owner)['sha256']=='c7cbef8662bbdf9a727c8e6d9f49a63ed11efe8c55e9859a06ffcbe48e4a68cb'
sp=importlib.util.spec_from_file_location('root859v25owner',owner);R=importlib.util.module_from_spec(sp);sys.modules[sp.name]=R;sp.loader.exec_module(R)
V=R.import_v24();R.checked(R.REPLAY,R.REPLAY_SHA,'v25 replay v2')
parent=H/'all-quality-854-root/round-02/verified-state.json'
state=H/'all-quality-859-root/state.json';config=state.parent/'context-config.json'
audit=H/'all-quality-859-audit775-root/proposal-exact-775/report.json'
out=H/'all-quality-859-qualified-root';out.mkdir(exist_ok=False)
assert json.loads(audit.read_text())['status']==R.ZERO_STATUS
replay=out/'independent-allrole-replay.json'
cmd=[sys.executable,str(R.REPLAY),'--candidate-state',str(state),'--parent-state',str(parent),'--audit-report',str(audit),'--audit-config',str(config),'--output-report',str(replay)]
R.run('independent-v25-replay',cmd,out/'replay-logs')
d=json.loads(replay.read_text());assert d['status']=='PASS_775_SOURCE_CONTEXT_VOLUME_QUALITY_AND_V25_FREE_SET'
base=replay.with_name(replay.stem+'-base-volume-replay.json')
dest=out/'quality-working-state.json'
V.finalize_verified_state(state,audit,base,dest)
final=json.loads(dest.read_text())
final.update(independent_allrole_local_replay=pin(replay),physical_admission=False,native_steps=0,mass_reconciliation={'status':'PENDING_VOLUME_CHANGE_RECONCILIATION','mass_conservation_claimed':False},quality_working_state_qualifier=pin(__file__))
dest.write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')
proof=V.verify_source_input(dest);assert str(proof['status']).startswith('PASS_')
(out/'source-proof-807.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
receipt={'status':'PASS_SOURCE_AND_NATIVE_CELL_GEOMETRY_ONLY','state':pin(dest),'parent':pin(parent),'proposal':pin(state),'audit775':pin(audit),'independent_replay':pin(replay),'driver':pin(__file__),'Skin':final['Skin_current']['native_quality_failures'],'Fat':final['Fat_current']['native_quality_failures'],'physical_admission':False,'native_steps':0,'mass_reconciliation':'PENDING'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt))
