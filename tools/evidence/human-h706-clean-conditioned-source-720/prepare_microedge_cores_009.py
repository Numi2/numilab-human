from pathlib import Path
import subprocess,json,time,hashlib
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209");H=P/"fine-physical-volume-706";F=P/"full-body-preparation-703"
PY="/Users/numi/numi-human-prep-venv-20261005/bin/python"
out=H/"microedge-cores-009"
if out.exists():raise FileExistsError(out)
out.mkdir()
jobs=[
 ("I",H/"local-microedge-condition-I-retry-001/candidate-I-microcollapsed.npz","fee20f3d7a1697158398c73a8a771302625e122b52ee23840101c552e3832205",
  P/"canonical-I328-lineage-356/I-temporary-cap-source.npz","f28553281c2c86cc91c433aed944c7c1b4635da0caa109b0873717618355c977"),
 ("D",H/"local-microedge-condition-D-retry-007/candidate-D-microcollapsed.npz","5503033cf67b3adb6533abc7711f2cde5a612fe4275833ef15b8ae1aba202115",
  P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/typed-D-62000100/run-002/D-temporary-cap-source-62000101.npz","5f5c622a67888243925ebfdc33c8f2c57109b0e4327a9386fcbc185ff72c1634")]
result={"schema":"numi.paired-fullbody-tetgen-readiness-697.v1","physical_admission":False,"runtime":0,"accepted_steps":0,"gates":{"positive_core_seeds":"NOT_RUN"},"jobs":[]}
for role,surface,digest,cap,cdigest in jobs:
 cmd=[PY,str(F/"prepare_registered_cavity_cores.py"),"--surface",str(surface),"--surface-sha256",digest,"--cap",str(cap),"--cap-sha256",cdigest,"--output",str(out/role)]
 start=time.monotonic()
 with (out/(role+".stdout.log")).open("w") as so,(out/(role+".stderr.log")).open("w") as se:r=subprocess.run(cmd,stdout=so,stderr=se)
 result["jobs"].append({"role":role,"argv":cmd,"exit_code":r.returncode,"wall_seconds":time.monotonic()-start})
 result["status"]="RUNNING" if r.returncode==0 else "FAIL_CORE_PREPARATION"
 (out/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result["jobs"][-1]),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
result["status"]="PASS_CONSTRUCTION_CORE_AND_OWNER_HOLE_EXTRACTION_ONLY";result["gates"]["positive_core_seeds"]="PASS"
for item in result["jobs"]:
 p=out/item["role"]/"report.json";item["report_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":result["status"],"physical_admission":False}),flush=True)
