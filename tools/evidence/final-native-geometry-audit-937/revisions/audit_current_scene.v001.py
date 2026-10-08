#!/usr/bin/env python3
"""Run the retained 908 exact geometry audit on a completed current owner scene."""
import argparse
import hashlib
import importlib.util
import json
import mmap
import re
import struct
import subprocess
import sys
from pathlib import Path

E = Path("/Users/n/numi-human-resting-evidence-20261005")
ROOT = Path("/Users/n/numi-human-resting-final-integration-001")
BASE = E / "native-common-skin-combined-cycle-audit-908"
PINS = {
    BASE / "audit_cycle.py": "2eba147ca37ea80a3ed12c8dd725986d77bcd60c194077fae6ddfc5961d9dcde",
    BASE / "verify_nha_topology.py": "18ba92b26acb48b5485ae5b1b3c1b0057e3549969d4bfeb32076ae5218c435b8",
    E / "cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py": "eb9e5c762cdbab1e9b3f98c63a1580a21acba54637b105c28e2bc0a39187ba17",
    ROOT / "src/numilab_human/common_atlas_skin_clearance.py": "1321c31c22e1b1dbf0062947c6767c2cfcae9aae35d29e9ea775862feed5ddd1",
    ROOT / "src/numilab_human/cardiac_cavity_intersections.py": "11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb",
}
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""): h.update(block)
    return h.hexdigest()

def writej(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n")

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def option(argv, key):
    if argv.count(key) != 1 or argv.index(key)+1 >= len(argv):
        raise ValueError("missing/duplicate argument: "+key)
    return argv[argv.index(key)+1]

def validate_documents(inv, meta, native_text):
    if meta.get("exit_code") != 0 or meta.get("source_files_changed_during_run") != []:
        raise ValueError("owner run is incomplete, failed, or changed inputs")
    for k in ("argv", "asset_sha256", "environment"):
        if meta.get(k) != inv.get(k) or not isinstance(inv.get(k), (dict, list)):
            raise ValueError("owner invocation/terminal disagreement: "+k)
    argv = inv["argv"]
    n = int(option(argv, "--muscle-step-count"))
    dt = float(option(argv, "--muscle-step-seconds"))
    if n <= 0 or dt != .002:
        raise ValueError("positive root count and recorded 2 ms timestep required")
    f32dt = struct.unpack("<f", struct.pack("<f", dt))[0]
    lines = re.findall(r"^resting_integrated_body=completed simulated_s=(\S+) wall_s=(\S+) real_time_factor=(\S+) physiology_body_clock=matched root_assistance=false .*?$", native_text, re.M)
    if len(lines) != 1 or abs(float(lines[0][0])-n*f32dt) > 1e-9:
        raise ValueError("missing exact unassisted accepted native completion")
    text = inv["environment"].get("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS", "")
    steps = [int(x) for x in text.split(",") if x]
    if not 1 <= len(steps) <= 8 or len(set(steps)) != len(steps):
        raise ValueError("one to eight distinct recorded capture IDs required")
    if any(s < 0 or s > n or (s not in (0,n) and (s+1)%32) for s in steps):
        raise ValueError("capture is outside compiled initial/submission/terminal cadence")
    runtime = meta.get("loaded_metal_runtime", {})
    if runtime.get("verified") is not True:
        raise ValueError("owner loaded-runtime verification absent")
    return steps, n, f32dt

def preflight(run):
    for p, expected in PINS.items():
        if sha(p) != expected: raise ValueError("retained helper changed: "+str(p))
    inv = json.loads((run/"invocation.json").read_text())
    meta = json.loads((run/"run-metadata.json").read_text())
    log = run/"native.log"
    steps, n, dt = validate_documents(inv, meta, log.read_text())
    assets = inv["asset_sha256"]
    for p, expected in assets.items():
        if sha(p) != expected: raise ValueError("owner asset changed: "+p)
    skin = Path(option(inv["argv"], "--skin-payload"))
    nha = Path(option(inv["argv"], "--torso-anatomy-payload"))
    receipt = Path(option(inv["argv"], "--resting-anatomy-receipt"))
    for p in (skin,nha,receipt):
        if str(p) not in assets: raise ValueError("unhashed scene argument: "+str(p))
    anatomy = json.loads(receipt.read_text())
    if anatomy.get("functional_bindings",{}).get("anatomy_payload_sha256") != assets[str(nha)]:
        raise ValueError("anatomy receipt is not bound to current source")
    for step in steps:
        p = run/f"accepted-geometry/step-{step}.mrvpack"
        r = json.loads(p.with_suffix(".receipt.json").read_text())
        if (r.get("pack_file_sha256") != sha(p) or r.get("accepted_step") != step
                or r.get("accepted_pack_path") != str(p)
                or r.get("physical_endpoint") != "accepted" or r.get("surface_audit_endpoint") != "passed"
                or r.get("common_field_source_anatomy_payload_sha256") != assets[str(nha)]
                or abs(r.get("accepted_time_s",-1)-step*dt) > 1e-9):
            raise ValueError("accepted pack receipt mismatch: "+str(p))
    module = load(BASE/"audit_cycle.py", "audit908")
    module.SKIN = skin
    module.NHA_PATH = nha
    module.NHA_SHA256 = assets[str(nha)]
    # run_step uses these legacy names ONLY as immutable input-hash entries.
    # Current provenance is the actual owner anatomy receipt, not a 907 claim.
    module.COMPOSITION_REPORT = receipt
    module.REGISTRATION_MANIFEST = receipt
    return module, steps, inv, meta

def topology(module, run, step):
    import numpy as np
    sys.path.insert(0,str(ROOT/"src"))
    from numilab_human import common_atlas_skin_clearance as c
    t = load(BASE/"verify_nha_topology.py", "topology908")
    _, _, keys = t.inventory()
    positions, surfaces, _ = c._pack_surfaces(run/f"accepted-geometry/step-{step}.mrvpack", keys)
    del positions
    results = []
    with module.NHA_PATH.open("rb") as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        rows, info = t.parse_nha(mm)
        if set(rows) != {sid for _,sid in keys}: raise ValueError("NHA identity set mismatch")
        for semantic, sid in sorted(keys):
            src = rows[sid]
            face = np.frombuffer(mm,dtype="<u4",count=src["index_count"],
                   offset=info["index_offset"]+src["first_index"]*4).reshape(-1,3).astype(np.int64)
            face -= src["first_vertex"]
            captured = surfaces[(semantic,sid)]["faces"]
            if captured.shape != face.shape: raise ValueError("NHA face count mismatch")
            delta = captured.astype(np.int64)-face
            exact = bool(delta.size and np.all(delta == delta.flat[0]) and delta.flat[0]>=0)
            if not exact: raise ValueError("NHA topology mismatch: "+str(sid))
            results.append({"semantic":semantic,"stable_id":sid,"source_face_count":len(face),
                            "index_order_match_under_uniform_pack_vertex_offset":True})
    return {"schema":"numi.human.accepted-nha-face-order-validation.v1","step":step,
            "all_524_nha_targets_match_current_source":len(results)==524,"surfaces":results,
            "qualification":"source identity and topology only; no geometric clearance claim"}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--preflight-only",action="store_true")
    ap.add_argument("--step",type=int,help=argparse.SUPPRESS)
    args=ap.parse_args()
    run=args.run.resolve();out=args.out.resolve()
    module,steps,inv,meta=preflight(run)
    if args.step is not None:
        if args.step not in steps: raise ValueError("unrecorded capture requested")
        writej(out/f"step-{args.step}.topology.json",topology(module,run,args.step))
        module.run_step(args.step,run,out)
        return 0
    if E not in out.parents or out.exists(): raise ValueError("fresh evidence output required")
    out.mkdir(parents=True)
    tracked=list(PINS)+[Path(__file__).resolve(),run/"invocation.json",run/"run-metadata.json",run/"native.log"]
    before={str(p):sha(p) for p in tracked}
    writej(out/"declaration.json",{"schema":"numi.human.native-combined-skin-cycle-audit.declaration.v1",
        "run_path":str(run),"steps":steps,"source_sha256":before,"skin_payload_path":str(module.SKIN),
        "skin_payload_sha256":sha(module.SKIN),"current_nha_payload_path":str(module.NHA_PATH),
        "current_nha_payload_sha256":module.NHA_SHA256,"surface_inventory_count":859,
        "qualification":"retained908 exact skin/self/all-target audit; current source topology; sparse captured geometry only",
        "preflight_only":args.preflight_only})
    if args.preflight_only:
        for step in steps: writej(out/f"step-{step}.topology.json",topology(module,run,step))
        writej(out/"preflight.json",{"status":"pass","steps":steps,"geometry_intersections_not_run":True})
        return 0
    with (out/"run.log").open("x") as log:
        for step in steps:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--run",str(run),
               "--out",str(out),"--step",str(step)],stdout=log,stderr=subprocess.STDOUT)
            if p.returncode: raise RuntimeError(f"step {step} failed; partial evidence retained")
    if before != {str(p):sha(p) for p in tracked}: raise ValueError("audit inputs changed")
    rows=[json.loads((out/f"step-{s}.result.json").read_text()) for s in steps]
    complete=all(r["pair_coverage_complete"] for r in rows)
    clear=complete and all(r["all_skin_crossing_pair_count"]==0 and r["skin_self_crossing_pair_count"]==0 for r in rows)
    writej(out/"summary.json",{"schema":"numi.human.native-combined-skin-cycle-audit.summary.v1",
        "status":"complete_pair_coverage" if complete else "incomplete_degenerate_input_fail_closed",
        "run_path":str(run),"skin_payload_sha256":sha(module.SKIN),"steps":rows,
        "all_requested_steps_complete":len(rows)==len(steps),
        "all_target_and_skin_self_pair_coverage_complete":complete,"intersection_free":clear,
        "qualification":"exact skin/self/all-target captured geometry only; completeness distinct from clearance; no full-body all-pair or continuous-time proof"})
    return 0 if clear else 2

if __name__=="__main__":
    raise SystemExit(main())
