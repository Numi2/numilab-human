from pathlib import Path
import json,hashlib,sys
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");p=H/"checkpoint.json"
s=Path(sys.argv[1]);expected=sys.argv[2];h=hashlib.sha256(s.read_bytes()).hexdigest();assert h==expected
z=json.loads(s.read_text());assert z["physical_admission"] is False and z["native_steps"]==0 and z["surface_crossings"]==0
d=json.loads(p.read_text());r={"path":str(s.resolve()),"sha256":h,"Skin":int(z["Skin_current"]["native_quality_failures"]),"Fat":int(z["Fat_current"]["native_quality_failures"]),"surface_crossings":0,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
d["current_quality_working_state_20261031"]=r;d.setdefault("quality_repair_progress_20261031",{})["quality_working_counts"]=r
tmp=p.with_name("checkpoint.selected-quality.tmp.json");tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n");tmp.replace(p);print(json.dumps(r))

