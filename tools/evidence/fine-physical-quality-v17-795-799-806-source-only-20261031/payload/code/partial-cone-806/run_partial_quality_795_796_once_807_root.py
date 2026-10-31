#!/usr/bin/env python3
"""One bounded serialized partial-quality cone proposal plus independent promotion."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

H=Path(__file__).resolve().parents[1]
PRODUCER=H/"quality-partial-cavity-684-root/batch_interior_steiner_partial_quality_795_root.py"
PROMOTER=H/"quality-partial-cavity-684-root/independent_promoter_796_795_root.py"
SOURCE_VERIFIER=H/"verify_current_source_evidence_791_root.py"
EXPECTED_PRODUCER_SHA256="4e35df53981c2fedc0f03b268138ebf74378a09e0c55d1f7a64b8e09c813f3bf"
EXPECTED_PROMOTER_SHA256="320fcbb6f1eb8147c67a27474c4b23100e92bca060122b3dffc37aa851bbcfec"
EXPECTED_SOURCE_VERIFIER_SHA256="b4da6ac3509767e11f3df6bd712a9908adf6b9bd0c1efc2122b16ec15df04ed9"
PASS_ROLE_STATUS="PASS_INTERIOR_STEINER_PARTIAL_QUALITY_CONE_BATCH"

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()

def pin(path):
    p=Path(path)
    return {"path":str(p),"sha256":sha(p),"size_bytes":p.stat().st_size}

def imp(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError("cannot import pinned dependency: "+str(path))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def validate_dependencies():
    for p,want in ((PRODUCER,EXPECTED_PRODUCER_SHA256),(PROMOTER,EXPECTED_PROMOTER_SHA256),
                   (SOURCE_VERIFIER,EXPECTED_SOURCE_VERIFIER_SHA256)):
        got=sha(p)
        if got!=want:raise RuntimeError(f"dependency SHA mismatch: {p} {got}")
    producer=imp("producer_795_for_serialized_807",PRODUCER)
    promoter=imp("promoter_796_for_serialized_807",PROMOTER)
    if not callable(getattr(producer,"run_state",None)) or not callable(getattr(promoter,"verify_and_promote",None)):
        raise RuntimeError("pinned producer/promoter entry point missing")
    return {"producer":pin(PRODUCER),"promoter":pin(PROMOTER),"source_evidence_verifier":pin(SOURCE_VERIFIER)},producer,promoter

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seconds",type=float,default=90.0)
    ap.add_argument("--maximum-attempts",type=int,default=10000)
    ap.add_argument("--maximum-operations",type=int,default=1000)
    ap.add_argument("--maximum-old-cells",type=int,default=24)
    ap.add_argument("--maximum-points",type=int,default=192)
    ap.add_argument("--maximum-star-cells",type=int,default=128)
    ap.add_argument("--construction-margin",type=float,default=0.0,
                    help="retained for producer argument compatibility; partial-quality helper uses exact visibility positivity")
    args=ap.parse_args()
    if args.seconds<=0 or args.maximum_attempts<1 or args.maximum_operations<1:
        ap.error("seconds, maximum attempts, and maximum operations must be positive")
    if not args.state.is_file():ap.error("--state must name an existing file")
    deps,producer,promoter=validate_dependencies()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    proposal=out/"proposal"
    producer_args=argparse.Namespace(maximum_old_cells=args.maximum_old_cells,
        maximum_points=args.maximum_points,maximum_star_cells=args.maximum_star_cells,
        maximum_attempts=args.maximum_attempts,maximum_operations=args.maximum_operations,
        construction_margin=args.construction_margin)
    started=time.monotonic()
    report=producer.run_state(args.state,proposal,args.seconds,producer_args)
    report_path=proposal/"report.json"
    role_progress=[r for r in report.get("role_results",[]) if r.get("status")==PASS_ROLE_STATUS]
    if not role_progress:
        result={"schema":"h706.partial-quality-steiner-single-round-807.v1",
            "status":"NO_VALID_BOUNDED_PARTIAL_QUALITY_PROGRESS","dependencies":deps,
            "source_state":pin(args.state),"producer_report":pin(report_path),
            "role_results":report.get("role_results",[]),"verified_state":None,
            "physical_admission":False,"native_steps":0,"elapsed_s":time.monotonic()-started}
        receipt=out/"run-receipt.json"
        receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"status":result["status"],"receipt":pin(receipt),
                          "physical_admission":False,"native_steps":0},sort_keys=True))
        return
    verified=out/"verified-state.json"
    promoter_receipt=out/"independent-promoter.json"
    promotion=promoter.verify_and_promote(args.state,report_path,verified,promoter_receipt)
    result={"schema":"h706.partial-quality-steiner-single-round-807.v1",
        "status":"PASS_INDEPENDENT_PARTIAL_QUALITY_PROMOTION",
        "dependencies":deps,"source_state":pin(args.state),
        "producer_report":pin(report_path),"producer_role_results":report.get("role_results",[]),
        "independent_promoter":promotion,"promoter_receipt":pin(promoter_receipt),
        "verified_state":pin(verified),"physical_admission":False,"native_steps":0,
        "mass_mapping":"NOT_RUN","elapsed_s":time.monotonic()-started}
    receipt=out/"run-receipt.json"
    receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"receipt":pin(receipt),
        "verified_state":pin(verified),
        "role_bad_before_after":[(r.get("role"),r.get("native_bad_before"),r.get("native_bad_after"),
            r.get("log_deficit_before"),r.get("log_deficit_after")) for r in role_progress],
        "physical_admission":False,"native_steps":0},sort_keys=True))

if __name__=="__main__":main()
