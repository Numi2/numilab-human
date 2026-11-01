#!/usr/bin/env python3
"""Fail-closed artifact-staging regression for v871."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys

PRODUCER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/quality-partial-cavity-684-root/paired_connected_boundary_patch_batch_871_replay_localids_staged_root.py")
PRODUCER_SHA="8faaf716c17bac710a0ceed0ea836bd10c63b11148e21a953eb3b2a0758be21e"

def pin(path):
    path=Path(path).resolve()
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError("control output must be fresh")
    if hashlib.sha256(PRODUCER.read_bytes()).hexdigest()!=PRODUCER_SHA:
        raise RuntimeError("pinned v871 producer hash mismatch")
    spec=importlib.util.spec_from_file_location("v871_staging_control",PRODUCER)
    if not spec or not spec.loader:raise RuntimeError("cannot import v871 producer")
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    def forced_failure(*a,**k):raise RuntimeError("synthetic independent replay failure control")
    m.independent_batch_replay=forced_failure
    try:m.paired_opposite_winding_tie_fixture(out)
    except RuntimeError as exc:
        if str(exc)!="synthetic independent replay failure control":raise
    else:raise AssertionError("forced independent replay failure did not propagate")
    run=out/"paired-I-run"
    required=[run/"candidate-I-typed-current.npz",run/"skin-batch-volume-proposal.npz",
      run/"fat-batch-volume-proposal.npz",run/"patch-journal.json",
      run/"pending-independent-replay.json",run/"independent-replay-failure.json"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise AssertionError("pre-replay artifacts missing: "+repr(missing))
    state=run/"state.json"
    if state.exists():raise AssertionError("verified state must not be written after replay failure")
    pending=json.loads((run/"pending-independent-replay.json").read_text())
    failure=json.loads((run/"independent-replay-failure.json").read_text())
    if pending.get("status")!="UNVERIFIED_PENDING_INDEPENDENT_REPLAY" or pending.get("verified_state_written") is not False:
        raise AssertionError("pending status not explicitly unverified")
    if failure.get("status")!="INDEPENDENT_REPLAY_FAILED" or failure.get("verified_state_written") is not False:
        raise AssertionError("failure status not fail-closed")
    if failure.get("exception")!="synthetic independent replay failure control":
        raise AssertionError("failure receipt does not preserve injected witness")
    receipt={"schema":"h706.v871-replay-failure-staging-control.v1","status":"PASS",
      "producer":pin(PRODUCER),"forced_failure":"synthetic independent replay failure control",
      "staged_artifacts":[pin(p) for p in required],"state_json_absent":True,
      "pending_status":"UNVERIFIED_PENDING_INDEPENDENT_REPLAY",
      "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
    receipt_path=out/"control-receipt.json"
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({**receipt,"control_receipt":pin(receipt_path)},sort_keys=True))

if __name__=="__main__":main()

