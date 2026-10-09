from pathlib import Path
import json,subprocess,time,hashlib,sys
R=Path("/Users/n/numi-human-retained-delivery-20261009");E=Path("/Users/n/numi-human-resting-evidence-20261005")
OUT=Path(__file__).parent;STUDY=R/"native-integrated-resting-study-1170"
PY="/Users/n/numi-human-prep-venv-20261005/bin/python3.13"
group=sys.argv[1]
if group=="analysis":
 jobs=[
 ("pair-report",[PY,str(OUT/"analyze_final_pair.py"),"--study",str(STUDY),"--output",str(R/"native-final-pair-report-1170-revision-003")]),
 ("descriptive-v004",[PY,str(E/"native-postrun-descriptive-supplement-1174/revision-004/postrun_descriptive_supplement.py"),"--study",str(STUDY),"--output",str(E/"native-postrun-descriptive-supplement-1174/revision-004/completed-pair-1170")])]
elif group=="media":
 base=R/"native-integrated-resting-study-1170-review/revision-002"
 jobs=[(arm+"-recording",[PY,str(base/"review_registered_arm_1174.py"),"--arm",arm,"--review-root",str(base/"results")]) for arm in ("control","treatment")]
else:raise ValueError(group)
failed=False
for label,argv in jobs:
 log=OUT/(label+".log");receipt=OUT/(label+".json")
 assert not log.exists() and not receipt.exists()
 start=time.time();clock=time.monotonic()
 with log.open("x") as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 d={"argv":argv,"started_unix":start,"finished_unix":time.time(),"wall_seconds":time.monotonic()-clock,"returncode":p.returncode,"log":str(log),"log_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),"script_sha256":hashlib.sha256(Path(argv[1]).read_bytes()).hexdigest()}
 receipt.write_text(json.dumps(d,indent=2)+"\n");print(json.dumps(d),flush=True)
 failed|=p.returncode!=0
sys.exit(1 if failed else 0)
