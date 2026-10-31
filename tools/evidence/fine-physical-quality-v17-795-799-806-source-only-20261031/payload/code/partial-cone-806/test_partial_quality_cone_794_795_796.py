#!/usr/bin/env python3
"""Small pinned end-to-end controls for the 794/795/796 partial-quality cone path."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
HELPER=H/"quality-kernel-cone-partial-794-root/interior_kernel_cone_partial_quality_794.py"
PRODUCER=H/"quality-partial-cavity-684-root/batch_interior_steiner_partial_quality_795_root.py"
PROMOTER=H/"quality-partial-cavity-684-root/independent_promoter_796_795_root.py"
PINS={
 "helper":"aa223d2b1b85d4ddefa1699deaee26be8cea96001b55f662df7388ac7a160495",
 "producer":"4e35df53981c2fedc0f03b268138ebf74378a09e0c55d1f7a64b8e09c813f3bf",
 "promoter":"320fcbb6f1eb8147c67a27474c4b23100e92bca060122b3dffc37aa851bbcfec",
}
def sha(p):return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def pin(p):return {"path":str(Path(p)),"sha256":sha(p),"size_bytes":Path(p).stat().st_size}
def run(cmd):
    x=subprocess.run(cmd,text=True,capture_output=True,check=True)
    try:return json.loads(x.stdout)
    except Exception as exc:raise RuntimeError(f"command did not emit JSON: {cmd}: {x.stdout[-500:]}") from exc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-root",type=Path,required=True)
    args=ap.parse_args()
    out=args.output_root
    out.mkdir(parents=True,exist_ok=False)
    for name,p in (("helper",HELPER),("producer",PRODUCER),("promoter",PROMOTER)):
        got=sha(p)
        if got!=PINS[name]:raise RuntimeError(f"{name} source pin mismatch: {got}")
    helper=run([sys.executable,str(HELPER)])
    if helper.get("status")!="PASS_EXACT_POSITIVE_PARTIAL_CONE_2_TO_1" or helper.get("old_bad")!=2 or helper.get("new_bad")!=1:
        raise RuntimeError("helper 2-to-1 exact-positive control failed")
    if helper.get("oriented_chain_equal") is not True or helper.get("exact_signed_volume_equal") is not True:
        raise RuntimeError("helper exact chain/volume control failed")
    prod_dir=out/"producer"
    prod=run([sys.executable,str(PRODUCER),"--fixture","--output",str(prod_dir)])
    producer_receipt=json.loads((prod_dir/"fixture-report.json").read_text())
    partial=producer_receipt.get("partial_quality_progress_control",{})
    if (prod.get("status")!="PASS_INTERIOR_STEINER_BATCH_SHARED_AND_PARTIAL_QUALITY_FIXTURES" or
        partial.get("status")!="PASS_PARTIAL_QUALITY_HELPER_2_TO_1_AND_BATCH_LEX_PROGRESS" or
        partial.get("all_new_cells_native_pass") is not False):
        raise RuntimeError("producer partial-quality 2-to-1 / partial-cone controls failed")
    prom_dir=out/"promoter"
    prom=run([sys.executable,str(PROMOTER),"--fixture","--output",str(prom_dir)])
    p=prom.get("partial_quality_progress",{})
    replay=p.get("independent_replay",{})
    if (prom.get("status")!="PASS_EXTERNAL_PROMOTER_NEGATIVE_CONTROLS_AND_MISSING_PROOF_REJECTION" or
        p.get("status")!="PASS_INDEPENDENT_PROMOTER_ACCEPTED_PARTIAL_CONE" or
        replay.get("status")!="PASS_EXTERNAL_INDEPENDENT_REPLAY" or
        replay.get("strict_quality_lexicographic_progress") is not True or
        not p.get("operation_readback",{}).get("all_output_cells_positive",False) or
        prom.get("all_four_mutations_rejected") is not True):
        raise RuntimeError("independent promoter partial-quality control failed")
    report={
      "schema":"h706.partial-quality-steiner-794-796-end-to-end-controls.v1",
      "status":"PASS_HELPER_PRODUCER_INDEPENDENT_PROMOTER_PARTIAL_QUALITY_CONTROLS",
      "source_pins":{n:pin(p) for n,p in (("helper",HELPER),("producer",PRODUCER),("promoter",PROMOTER))},
      "helper_fixture":helper,
      "producer_fixture":{"report":pin(prod_dir/"fixture-report.json"),"partial_quality":partial},
      "promoter_fixture":{"stdout_status":prom.get("status"),
          "partial_quality":p,"negative_controls_rejected":prom.get("all_four_mutations_rejected")},
      "method_limits":{"actual_geometry_run":False,"native_steps":0,"physical_admission":False,
          "partial_quality_proposal_only":True,"all_replacement_tets_exact_positive":True,
          "native_quality_of_every_replacement_tet":False,
          "fixed_boundary_chain_and_exact_signed_volume":True},
      "elapsed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
    }
    report_path=out/"test-receipt.json"
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"receipt":pin(report_path),
        "helper_bad":[helper["old_bad"],helper["new_bad"]],
        "producer_global_bad":[partial["producer_global_bad_before"],partial["producer_global_bad_after"]],
        "promoter_partial_status":p["status"],"physical_admission":False,"native_steps":0},sort_keys=True))
if __name__=="__main__":main()
