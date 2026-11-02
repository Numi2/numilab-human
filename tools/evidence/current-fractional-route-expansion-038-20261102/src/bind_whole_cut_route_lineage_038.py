#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

MANIFEST=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/postpass-030d-canonical-038/out-007/run-manifest.json")
RDIR=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/actual-038-cut-driven-augmentation-001/whole-cut-current-R-routes-v1")
BASE=RDIR.parent
A=BASE.parent
GEN=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/postpass-030d-canonical-038")
EXPECTED={
 "whole-cut-route-census-report.json":"d5d47d1410b0a5c0e3e91a29a3c1d51f9bb82e22647ffd82ce7c973d41726ffd",
 "whole-cut-crossing-routes.npz":"6ba416a2f06261e2e002b0727b1e0497ce447ddca6db7f7466092111ef5a4009",
 "enumerate_whole_cut_routes_038_v1.py":"cee02a011d27e1fad13f7c66b91aefa19a9bd8233f7e7b6836146695234ca069",
 "screen_cut_spare_routes_038_v6.py":"ae205ea5516771bbc24930c49ba290c1390e7f05f76684599d10f27519f10e94",
 "append_skin_witnesses_from_whole_cut_routes_038c.py":"e3694a708ac4ce1a91b2d3574ffb2dbb66ad69330d354daa45b1af9492768abd",
 "verify_and_finalize_skin_witness_append_038d.py":"4fdc7cc91ab733f604c292246558a8a110dab4b6de7d171950e33e5baf402978",
 "prepare_run_manifest_038b.py":"9089b62122ccab5270a3d084b84c571ee127fc6a5a105c3c1c14c36e4d82a24e",
 "prepare_run_manifest_038c.py":"ea830f1c9cd4211642c80c32cd7fc532ad0fa64eb85fadd2d8b426cd43575887",
 "whole-cut-route-skin-append-validation.json":"16087f0d6c53a2486623efa80c6a80cf2bad1a0bc40428407e27b1a35f779aa4",
}

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def pin(path:Path,expected:str|None=None)->dict:
 p=path.resolve(strict=True)
 if not p.is_file(): raise ValueError(f"not a file: {p}")
 digest=sha(p)
 if expected is not None and digest!=expected: raise ValueError(f"pinned source changed: {p}")
 return {"path":str(p),"sha256":digest,"bytes":p.stat().st_size}

def main():
 if not MANIFEST.is_file(): raise FileNotFoundError(MANIFEST)
 m=json.loads(MANIFEST.read_text())
 if m.get("schema")!="numi.h706.current-fractional-mass-cli-run.v1" or m.get("physical_admission") is not False or m.get("native_steps")!=0:
  raise ValueError("out-007 is not an unrun canonical source-only manifest")
 if "carrier_witness_skin_route_lineage" in m: raise ValueError("lineage already bound")
 named=m["named_input_pins"]
 out_skin=named["carrier_witness_skin"]
 append_report=pin(RDIR/"whole-cut-route-skin-append-validation.json",EXPECTED["whole-cut-route-skin-append-validation.json"])
 append=json.loads(Path(append_report["path"]).read_text())
 if append.get("status")!="PASS_ALL_ROUTE_SLOTS_REPLAYED_AND_APPENDED" or append.get("physical_admission") is not False or append.get("native_steps")!=0:
  raise ValueError("append replay is not source-only PASS")
 if append["output_csv"]["path"]!=out_skin["path"] or append["output_csv"]["sha256"]!=out_skin["sha256"] or append["output_csv"]["bytes"]!=out_skin["bytes"]:
  raise ValueError("run manifest Skin CSV differs from verified append output")
 census=json.loads((RDIR/"whole-cut-route-census-report.json").read_text())
 if census["proposal_artifact"]["sha256"]!=EXPECTED["whole-cut-crossing-routes.npz"]:
  raise ValueError("route census does not bind the expected proposal NPZ")
 refs=[]
 for name,digest in EXPECTED.items():
  if name in {"whole-cut-route-census-report.json","whole-cut-crossing-routes.npz","whole-cut-route-skin-append-validation.json","append_skin_witnesses_from_whole_cut_routes_038c.py","verify_and_finalize_skin_witness_append_038d.py"}:
   path=RDIR/name
  elif name in {"enumerate_whole_cut_routes_038_v1.py","screen_cut_spare_routes_038_v6.py"}:
   path=BASE/name
  else:
   path=GEN/name
  refs.append(pin(path,digest))
 # The recovered census depends on the exact full-cut artifacts; pin those too.
 for key in ("cut_gzip","cut_header","cut_manifest"):
  item=census["pins"][key]
  refs.append(pin(Path(item["path"]),item["sha256"]))
 refs.append(pin(Path(__file__).resolve()))
 existing={row["path"]:row for row in m["file_pins"]}
 for ref in refs:
  old=existing.get(ref["path"])
  if old is not None and old["sha256"]!=ref["sha256"]: raise ValueError(f"conflicting file pin: {ref['path']}")
  existing[ref["path"]]=ref
 m["file_pins"]=[existing[k] for k in sorted(existing)]
 m["carrier_witness_skin_route_lineage"]={
  "schema":"numi.h706.current-skin-witness-route-lineage.v1",
  "append_validation":append_report,
  "manifest_binding_source":pin(Path(__file__).resolve()),
  "route_census_report":pin(RDIR/"whole-cut-route-census-report.json",EXPECTED["whole-cut-route-census-report.json"]),
  "route_proposal_npz":pin(RDIR/"whole-cut-crossing-routes.npz",EXPECTED["whole-cut-crossing-routes.npz"]),
  "output_csv":out_skin,
  "route_pair_count":append["validation"]["route_pairs"],
  "current_T_skin_children_routed":append["validation"]["routed_T_skin_children"],
  "unrouted_T_skin_children":append["validation"]["unrouted_T_skin_children"],
  "physical_admission":False,"native_steps":0,
  "interpretation":"source-supported witness input expansion; no full-flow result or mass admission"
 }
 tmp=MANIFEST.with_suffix(".json.tmp")
 tmp.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
 os.replace(tmp,MANIFEST)
 print(json.dumps({"status":"PASS_ROUTE_LINEAGE_BOUND","manifest_path":str(MANIFEST),"file_pin_count":len(m["file_pins"]),"route_lineage":m["carrier_witness_skin_route_lineage"]},indent=2,sort_keys=True))

if __name__=="__main__": main()
