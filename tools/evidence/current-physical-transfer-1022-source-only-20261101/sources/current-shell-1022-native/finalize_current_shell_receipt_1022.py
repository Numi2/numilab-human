#!/usr/bin/env python3
"""Bind a completed 1022 shell receipt to the exact current O/I/D pins."""
from __future__ import annotations
import hashlib, importlib.util, json, os, shutil, tempfile
from pathlib import Path

H=Path(__file__).resolve().parent
BASE=H.parent
BRIDGE=BASE/"current-role-p1-support-938/current_fine_volume_to_217_938_v2_coordinate_alias_guard.py"
BRIDGE_SHA="1756c7126f708141b06b57e9b653a121d038d215d8801bb7e28f5220edcc8c66"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p).resolve()
 return {"path":str(p),"sha256":sha(p),"size_bytes":p.stat().st_size}
def main():
 import argparse
 ap=argparse.ArgumentParser()
 ap.add_argument("--receipt",type=Path,required=True)
 ap.add_argument("--state",type=Path,required=True)
 ap.add_argument("--report",type=Path,required=True)
 a=ap.parse_args()
 receipt_path=a.receipt.resolve();state_path=a.state.resolve();report_path=a.report.resolve()
 if report_path.exists():raise FileExistsError(report_path)
 r=json.loads(receipt_path.read_text()); state=json.loads(state_path.read_text())
 if r.get("physical_admission") is not False or r.get("native_steps")!=0:
  raise ValueError("shell receipt is not source-only")
 if r.get("current_state",{}).get("path")!=str(state_path) or r.get("current_state",{}).get("sha256")!=sha(state_path):
  raise ValueError("shell receipt does not bind the supplied current state")
 old_inputs=r.get("inputs",{})
 surfaces=old_inputs.get("source_surfaces")
 if not isinstance(surfaces,dict) or set(surfaces)!={"O","I","D"}:
  raise ValueError("initial shell receipt lacks paired current source surface pins")
 for role,item in surfaces.items():
  p=Path(item["path"])
  if not p.is_file() or sha(p)!=item["sha256"] or item!=state["source_surfaces"][role]:
   raise ValueError(f"{role} input does not match selected current state")
 build=old_inputs.get("runner")
 if not isinstance(build,dict):
  raise ValueError("executed shell producer pin is absent")
 archive=H/"source-freeze"
 archive.mkdir(exist_ok=True)
 frozen=archive/"build_current_shell_1022_run001.py"
 if frozen.exists() and sha(frozen)==build["sha256"]:
  pass
 elif Path(build["path"]).is_file() and sha(build["path"])==build["sha256"]:
  shutil.copy2(build["path"],frozen)
 else:
  raise ValueError("executed shell producer pin is not recoverable")
 if sha(frozen)!=build["sha256"]:raise ValueError("executed producer freeze differs from receipt pin")
 pre=receipt_path.with_name("receipt-prebind-001.json")
 if pre.exists():raise FileExistsError(pre)
 shutil.copy2(receipt_path,pre)
 inputs=dict(old_inputs)
 inputs["runner"] = pin(frozen)
 for role,item in surfaces.items():
  inputs[f"source_surface_{role}"] = item
 r["inputs"]=inputs
 r["receipt_binding_finalization"]={
  "kind":"top_level_938v2_source_pin_binding",
  "preserved_initial_receipt":pin(pre),
  "executed_producer_sha256":build["sha256"],
  "source_freeze":pin(frozen),
  "finalizer":pin(Path(__file__)),
  "reason":"938v2 receipt reader consumes named path/SHA pins at inputs top level"}
 tmp=receipt_path.with_suffix(".json.tmp")
 tmp.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
 os.replace(tmp,receipt_path)
 spec=importlib.util.spec_from_file_location("current_shell_bridge_finalize",BRIDGE)
 if spec is None or spec.loader is None:raise ImportError(str(BRIDGE))
 bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
 if sha(BRIDGE)!=BRIDGE_SHA:raise ValueError("pinned 938v2 bridge changed")
 bridge.load_shell_receipt(receipt_path,
  Path(r["outputs"]["skin-shell.npz"]["path"]),
  Path(r["outputs"]["fat-shell.npz"]["path"]),state)
 summary={
  "status":"PASS_938V2_CURRENT_SOURCE_SHELL_RECEIPT_BINDING",
  "receipt":pin(receipt_path),"preserved_initial_receipt":pin(pre),
  "state":pin(state_path),"source_surfaces":{k:pin(v["path"]) for k,v in surfaces.items()},
  "producer":pin(frozen),"physical_admission":False,"native_steps":0}
 report_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":summary["status"],"receipt":str(receipt_path),
                   "report":str(report_path),"source_freeze":str(frozen)},sort_keys=True))
if __name__=="__main__":main()
