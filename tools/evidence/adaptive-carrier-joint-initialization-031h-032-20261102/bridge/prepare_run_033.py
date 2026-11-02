#!/usr/bin/env python3
import copy, hashlib, json
from pathlib import Path

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
NATIVE=ROOT/"current-fractional-semantic-native-1047"
SOURCE_RUN=NATIVE/"actual-027-proof-slot-closure/run-manifest.json"
PACKET_BATCH=NATIVE/"optimization-trial-032-adaptive-joint-packet/run-004/packets/batch-packet-verification.json"
RUN=NATIVE/"actual-029-adaptive-joint-031-baseline-route/run-001"
PACKETS=NATIVE/"optimization-trial-032-adaptive-joint-packet/run-004/packets"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p);return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
def main():
 if RUN.exists() and any(RUN.iterdir()):raise SystemExit("refusing existing/nonempty actual run directory")
 old=json.loads(SOURCE_RUN.read_text());batch=json.loads(PACKET_BATCH.read_text())
 if batch.get("status")!="PASS_SOURCE_ONLY_ADAPTIVE_PACKET_READBACK" or batch.get("native_steps")!=0 or batch.get("physical_admission") is not False:raise SystemExit("packet batch is not source-only PASS")
 RUN.mkdir(parents=True,exist_ok=True)
 m=copy.deepcopy(old)
 m["base_manifest"]={"path":str(SOURCE_RUN),"sha256":sha(SOURCE_RUN)}
 m["prior_run_status"]="INFEASIBLE_FULL_GRAPH_CUT"
 m["adaptive_joint_proposal"]={
  "proposal_report":pin(ROOT/"current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-031-adaptive-anchor-joint-lp-001/run-007/proposal/joint-lp-report.json"),
  "proposal_npz":pin(ROOT/"current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-031-adaptive-anchor-joint-lp-001/run-007/proposal/adaptive-joint-weight-proposal.npz"),
  "matrix":pin(ROOT/"current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-028-complete-predicate-joint-lp-001/run-001/anchor-cut-matrix.csv"),
  "packet_batch":pin(PACKET_BATCH),
  "adapter_source":batch["bridge_source"],
  "scope":"source-only adaptive quadrature proposal with convex baseline-route retention; native allocator run required; historical cuts are not a full admission claim",
  "matrix_run_manifest":pin(ROOT/"current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-028-complete-predicate-joint-lp-001/run-001/run-manifest.json"),
  "matrix_execution":pin(ROOT/"current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-028-complete-predicate-joint-lp-001/run-001/execution.json")
 }
 m["baseline_route_blend"]=batch["baseline_route_blend"]
 m["input_source_run_manifest"]={"path":str(SOURCE_RUN),"sha256":sha(SOURCE_RUN)}
 m["named_input_pins"]=copy.deepcopy(old["named_input_pins"])
 for role in ("Skin","Fat"):
  key=role.lower()
  rec=batch["roles"][role]
  outputs=rec["outputs"]
  m["named_input_pins"]["carrier_"+key+"_packet"]=outputs["npz"]
  m["named_input_pins"]["carrier_witness_"+key]={"path":outputs["witness_csv"]["path"],"sha256":outputs["witness_csv"]["sha256"],"bytes":outputs["witness_csv"]["bytes"]}
 m["named_input_pins"]["adaptive_packet_batch"]=pin(PACKET_BATCH)
 m["file_pins"]=[x for x in m.get("file_pins",[]) if x["path"] not in {
  old["named_input_pins"]["carrier_skin_packet"]["path"],old["named_input_pins"]["carrier_fat_packet"]["path"],
  old["named_input_pins"]["carrier_witness_skin"]["path"],old["named_input_pins"]["carrier_witness_fat"]["path"]}]
 for role in ("Skin","Fat"):
  rec=batch["roles"][role]
  for name,obj in rec["outputs"].items():
   m["file_pins"].append({"path":obj["path"],"sha256":obj["sha256"],"bytes":obj["bytes"]})
 m["file_pins"].append(pin(PACKET_BATCH))
 argv=list(old["argv"])
 binpath=old["binary"]["path"]
 try:bi=argv.index(binpath)
 except ValueError:raise SystemExit("pinned CLI binary missing from old argv")
 env={}
 for token in argv[1:bi]:
  if "=" not in token:raise SystemExit("unexpected argv token before binary: "+token)
  k,v=token.split("=",1);env[k]=v
 env["NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ"]=m["named_input_pins"]["carrier_skin_packet"]["path"]
 env["NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256"]=m["named_input_pins"]["carrier_skin_packet"]["sha256"]
 env["NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ"]=m["named_input_pins"]["carrier_fat_packet"]["path"]
 env["NUMI_H706_FRACTIONAL_CARRIERS_FAT_SHA256"]=m["named_input_pins"]["carrier_fat_packet"]["sha256"]
 env["NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV"]=m["named_input_pins"]["carrier_witness_skin"]["path"]
 env["NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256"]=m["named_input_pins"]["carrier_witness_skin"]["sha256"]
 env["NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV"]=m["named_input_pins"]["carrier_witness_fat"]["path"]
 env["NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256"]=m["named_input_pins"]["carrier_witness_fat"]["sha256"]
 outdir=RUN/"verified-flow"
 if outdir.exists():raise SystemExit("refusing existing output directory")
 env["NUMI_H706_FRACTIONAL_OUTPUT_DIR"]=str(outdir)
 m["argv"]=["/usr/bin/env"]+[f"{k}={v}" for k,v in sorted(env.items())]+argv[bi:]
 m["stdout_log"]=str(RUN/"run.log")
 m["execution_json"]=str(RUN/"execution.json")
 m["cwd"]=str(NATIVE)
 m["physical_admission"]=False;m["native_steps"]=0
 m["experimental_status"]="READY_SOURCE_ONLY_ADAPTIVE_PACKET_RUN_WITH_BASELINE_ROUTE_BLEND"
 path=RUN/"run-manifest.json"
 path.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"path":str(path),"sha256":sha(path),"output_dir":str(outdir),"argv_count":len(m["argv"]),"input_pins":len(m["file_pins"])},sort_keys=True))
if __name__=="__main__":main()

