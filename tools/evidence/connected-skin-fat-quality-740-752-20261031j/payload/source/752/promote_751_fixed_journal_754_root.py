from pathlib import Path
import json,importlib.util,hashlib
h=Path(__file__).resolve().parent
v=h/'verify_fixed_boundary_journal_752_root.py'
spec=importlib.util.spec_from_file_location('v752',v);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
o=h/'all-quality-751-root/round001/flips';parent=h/'all-quality-751-root/round001/points/verified-state.json';r=m.verify_step(o/'report.json',o/'state.json',parent)
pin=lambda p:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rp=o/'independent-replay-752.json';rp.write_text(json.dumps({'status':'PASS_INDEPENDENT_FIXED_BOUNDARY_REPLAY','replay':r,'verifier':pin(v),'physical_admission':False,'native_steps':0},indent=2)+'\n')
st=json.load(open(o/'state.json'));st['independent_replay']=pin(rp);st['surface_crossings']=0;st['source_crossing_evidence_inherited_from_unchanged_surfaces']=pin(parent);vp=o/'verified-state.json';vp.write_text(json.dumps(st,indent=2)+'\n');print(json.dumps({'state':pin(vp),'counts':[(x,st[x+'_current']['native_quality_failures']) for x in ['Skin','Fat']]}))
