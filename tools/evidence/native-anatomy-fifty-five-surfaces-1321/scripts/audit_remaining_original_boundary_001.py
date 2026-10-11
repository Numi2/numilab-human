from pathlib import Path
import sys,json,re,subprocess,hashlib
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");sid=int(sys.argv[1]);D=A/f"remaining-original-boundary-batch-001/stable-{sid}"
lift=D/"source-lift-001";r=json.loads((lift/"report.json").read_text());assert r["complete"] and r["inputs_unchanged"] and r["rows"][0]["source_and_sampled_pose_self_clear"]
p=(A/"audit_stable127_original_split_010.py").read_text()
p=p.replace("stable-127-original-split-lift-009",str(lift.relative_to(A)))
p=p.replace('OUT=A/"stable-127-current48-independent-010"',f'OUT=Path({str(D/"current50-target-audit-001")!r})')
p=re.sub(r"\b127\b",str(sid),p).replace("stable127",f"stable{sid}")
p=p.replace("forty-eight-surface-native-composition-001","fifty-surface-native-composition-001").replace("forty-seven-surface-native-composition-001","forty-eight-surface-native-composition-001")
p=p.replace("current48","current50").replace("_row_arrays_identical_43_to_44","_row_arrays_identical_48_to_50")
p=p.replace("461444ae07be31b593995aa02b1fbced4088da49594fcc274fc21c7d3e072932",r["rows"][0]["candidate_sha256"])
script=D/"audit_current50_targets_001.py";assert not script.exists();script.write_text(p);compile(p,str(script),"exec")
with (D/"audit-current50-001.log").open("w") as log:result=subprocess.run([sys.executable,str(script)],stdout=log,stderr=subprocess.STDOUT)
print(sid,result.returncode)
