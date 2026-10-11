from pathlib import Path
import sys,json,subprocess,hashlib
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
sid=int(sys.argv[1]);D=A/f"remaining-original-boundary-batch-001/stable-{sid}";report=json.loads((D/"source-separation/report.json").read_text())
trial=next(x for x in report["trials"] if x.get("candidate_path"))
O=D/"source-lift-001";script=D/"lift_source_001.py";assert not script.exists()
p=(A/"lift_stable127_original_split_009.py").read_text()
p=p.replace("stable-127",f"stable-{sid}").replace("t,127",f"t,{sid}").replace("(127,t",f"({sid},t").replace('"stable_id":127',f'"stable_id":{sid}').replace('manifest_rows"][127]',f'manifest_rows"][{sid}]')
p=p.replace(f'A/"stable-{sid}-original-quotient-parent-004/stable-{sid}-original-quotient.npz"',f'Path({str(D/"original-quotient.npz")!r})')
p=p.replace(f'A/"stable-{sid}-original-quotient-parent-004/report.json"',f'Path({str(D/"source-separation/report.json")!r})')
p=p.replace("forty-eight-surface-native-composition-001","fifty-surface-native-composition-001")
p=p.replace('assert rr["pure_attachment_proxy_count"]==0',"""missing=[]
for j in np.unique(original["faces"]):
 if original["weights"][j].max()<.999999:continue
 hits=np.flatnonzero(np.all(pos==original["positions"][j],axis=1))
 if not any(np.array_equal(local[i],original["local"][j]) and np.array_equal(weights[i],original["weights"][j]) for i in hits):missing.append(int(j))
rr["changed_or_missing_pure_attachment_proxies"]=missing
assert not missing
""")
compile(p,str(script),"exec");script.write_text(p)
with (D/"lift-001.log").open("w") as log:
 result=subprocess.run([sys.executable,str(script),trial["candidate_path"],str(O)],stdout=log,stderr=subprocess.STDOUT)
print(sid,result.returncode)
