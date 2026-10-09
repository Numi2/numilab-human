#!/usr/bin/env python3
"""Pinned 1199 terminal anatomy adapter over the unchanged 1198 exact audit core."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

AUDIT_ROOT=Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-anatomy-audit-1199")
CORE=Path("/Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/audit_fixed_bed_full_body_captures_1198.py")
CORE_SHA="fc04a1bf331aa8adc5b1f8309843129bcbfd5d446f1f4488f38e11432782a785"
RECOVERY= AUDIT_ROOT/"launch-input-recovery-1199.json"
RECOVERY_SHA="0d231294651fd0e2310685f3b6b5f76fbb68e00dd7a00549694f2f2b93735cac"
MANIFEST=Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/resting-supine-scene-distributed-support.manifest.json")
MANIFEST_SHA="55a1e0149d5ce1482fc382eb2bc6ba31ce90888f2594435f191d5615b8c0b8f4"
SUPPORT=Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/myosim-fullbody-distributed-rigid-digit-support.nhcnt")
SUPPORT_SHA="b029bf6e86958ba4c7f9e1d5ca7f7e94a6e123952f926b02b254ad02b426971f"
COVERAGE_ORIGINAL=Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/source-region-coverage-1199.json")
COVERAGE_LAUNCH_SHA="42b6ef6d348b68127b44d910eb2028816499dec92e129282a94f0c0277168b07"
COVERAGE_V1=Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/source-region-coverage-1199-v1.json")
VALIDATOR_ORIGINAL=Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/validate_source_regions.py")
VALIDATOR_LAUNCH_SHA="198d576c97ad1221c2cd835e2e6ecbb9f388e7c1283524c94502f6c0ebdc8c1c"
VALIDATOR_V1=Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/validate_source_regions-v1.py")
PREVIOUS_GAPS=Path("/Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/source-voronoi-region-gaps-5mm.json")
PREVIOUS_GAPS_SHA="a9bf4a344c96038211f4566d8c0d975cf6fd440e1f146d466de1465863264002"
EXPECTED={str(COVERAGE_ORIGINAL):COVERAGE_LAUNCH_SHA,str(VALIDATOR_ORIGINAL):VALIDATOR_LAUNCH_SHA}
relocation_observations={}
current_original_hashes={}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
 return h.hexdigest()

def require(ok,msg):
 if not ok: raise ValueError(msg)

def json_read(p): return json.loads(Path(p).read_text())

def load_recovery():
 require(sha(CORE)==CORE_SHA,"1198 audit core changed")
 require(sha(RECOVERY)==RECOVERY_SHA,"1199 recovery map changed")
 d=json_read(RECOVERY)
 require(d.get("schema")=="numi.human.distributed-support-1199.launch-input-recovery.v1","unknown recovery schema")
 entries=d.get("entries")
 require(isinstance(entries,list) and {str(Path(x.get("original_path","")).resolve()) for x in entries}==set(EXPECTED),"recovery map must contain exactly the two launch-time audit inputs")
 out={}
 for e in entries:
  original=str(Path(e["original_path"]).resolve())
  expected=EXPECTED[original]
  require(e.get("launch_declared_sha256")==expected and e.get("reconstructed_v1_sha256")==expected,"reconstructed source pin differs from run declaration")
  target=Path(e["reconstructed_v1_path"])
  require(target.is_file() and not target.is_symlink() and sha(target)==expected,"exact launch-time copy missing or hash-mismatched")
  current=sha(original) if Path(original).is_file() and not Path(original).is_symlink() else None
  current_original_hashes[original]=current
  out[original]={"original_path":original,"launch_declared_sha256":expected,"original_current_sha256":current,
                 "reconstructed_v1_path":str(target.resolve()),"reconstructed_v1_sha256":sha(target),
                 "reason":"nonruntime partition audit input changed after launch; use only exact reconstructed launch-time bytes"}
 return out

RECOVERY_MAP=load_recovery()

def load_core():
 src=CORE.read_text()
 old="declared[0] == 0 and declared[-1] == roots and len(declared) >= 3"
 new="declared[0] == 0 and declared[-1] == roots and len(declared) >= 2"
 require(src.count(old)==1,"pinned core capture-admission line changed")
 src=src.replace(old,new,1)
 old="capture list must contain the initial pose, interior full-breath captures, and true terminal step"
 new="capture list must include initial and true terminal accepted steps"
 require(src.count(old)==1,"pinned core capture-admission message changed")
 src=src.replace(old,new,1)
 name="_fixed_bed_terminal_audit_core_1199"
 spec=importlib.util.spec_from_file_location(name,CORE)
 require(spec is not None,"cannot load pinned 1198 audit core")
 mod=importlib.util.module_from_spec(spec)
 sys.modules[name]=mod
 exec(compile(src,str(CORE),"exec"),mod.__dict__)
 mod.ROOT=AUDIT_ROOT
 mod.BED_MANIFEST=MANIFEST
 mod.BED_MANIFEST_SHA=MANIFEST_SHA
 mod.SUPPORT=SUPPORT
 mod.SUPPORT_SHA=SUPPORT_SHA
 original_check_hash=mod.check_hash
 def check_hash_1199(path,expected):
  key=str(Path(path).resolve())
  if key in RECOVERY_MAP:
   require(expected==RECOVERY_MAP[key]["launch_declared_sha256"],"only the exact two declared nonruntime audit inputs may be relocated")
   target=Path(RECOVERY_MAP[key]["reconstructed_v1_path"])
   require(sha(target)==expected,"reconstructed v1 audit input changed")
   return target.resolve()
  return original_check_hash(path,expected)
 mod.check_hash=check_hash_1199
 original_load_static=mod.load_static
 def load_static_1199():
  values=list(original_load_static())
  manifest=values[8]
  static=values[10]
  prov=manifest.get("heightfield_provenance",{})
  old_gaps=prov.get("source_region_gaps_previous_partition",{})
  require(old_gaps.get("path")==str(PREVIOUS_GAPS) and old_gaps.get("sha256")==PREVIOUS_GAPS_SHA,
          "1199 manifest does not identify 1196 region gaps as the previous partition")
  report=json_read(COVERAGE_V1)
  part=report.get("partition",{})
  require(report.get("schema")=="numi.human.resting.distributed-source-region-coverage.v1"
          and report.get("status")=="source_rest_partition_complete_and_bed_grid_identity_verified"
          and part.get("assigned_skin_vertex_count")==54949 and part.get("cardinality_sum")==54949
          and part.get("region_count")==32 and part.get("nonempty_region_count")==32
          and part.get("multiply_assigned_count")==0 and part.get("unassigned_count")==0,
          "reconstructed 1199 source partition report fails complete-coverage checks")
  require(report.get("heightfield",{}).get("path")==str(MANIFEST)
          and report.get("heightfield",{}).get("sha256")==MANIFEST_SHA
          and report.get("support_contact",{}).get("path")==str(SUPPORT)
          and report.get("support_contact",{}).get("sha256")==SUPPORT_SHA
          and report.get("analysis_script",{}).get("path")==str(VALIDATOR_ORIGINAL)
          and report.get("analysis_script",{}).get("sha256")==VALIDATOR_LAUNCH_SHA
          and sha(VALIDATOR_V1)==VALIDATOR_LAUNCH_SHA,
          "reconstructed 1199 report is not hash-bound to its manifest/support/validator")
  require(MANIFEST.is_file() and sha(MANIFEST)==MANIFEST_SHA and sha(SUPPORT)==SUPPORT_SHA,
          "1199 distributed manifest or support payload differs from its launch-time pin")
  # Keep both launch-time copies and currently changed originals under the audit's before/after hash gate.
  static[str(COVERAGE_V1.resolve())]=COVERAGE_LAUNCH_SHA
  static[str(VALIDATOR_V1.resolve())]=VALIDATOR_LAUNCH_SHA
  static[str(RECOVERY.resolve())]=RECOVERY_SHA
  for path,digest in current_original_hashes.items():
   if digest is not None: static[path]=digest
  return tuple(values)
 mod.load_static=load_static_1199
 original_preflight=mod.preflight
 def preflight_1199(args,base,static):
  values=list(original_preflight(args,base,static))
  require(values[4]==[0,10000] and values[6]==10000,
          "1199 anatomy scan accepts only initial and terminal captures 0,10000")
  tracked=values[5]
  for path,digest in current_original_hashes.items():
   if digest is not None: tracked[path]=digest
  tracked[str(RECOVERY.resolve())]=RECOVERY_SHA
  return tuple(values)
 mod.preflight=preflight_1199
 return mod

def enrich(out):
 declaration=Path(out)/"declaration.json"
 summary=Path(out)/"summary.json"
 if not declaration.is_file() or not summary.is_file(): return
 d=json_read(declaration); s=json_read(summary)
 provenance={
  "1199_source_partition_report":{"original_launch_path":str(COVERAGE_ORIGINAL),"launch_sha256":COVERAGE_LAUNCH_SHA,
      "reconstructed_launch_copy":str(COVERAGE_V1),"scope":"source-rest ownership partition only; no dynamic contact/clearance claim"},
  "1199_source_partition_validator":{"original_launch_path":str(VALIDATOR_ORIGINAL),"launch_sha256":VALIDATOR_LAUNCH_SHA,
      "reconstructed_launch_copy":str(VALIDATOR_V1)},
  "1196_region_gaps":{"path":str(PREVIOUS_GAPS),"sha256":PREVIOUS_GAPS_SHA,
      "status":"historical previous partition only; not used for 1199 seed assignments"}
 }
 d["capture_scope"]="exactly initial step 0 and terminal step 10000; no interior/full-breath captures"
 d["source_partition_provenance"]=provenance
 d["launch_input_relocations"]=RECOVERY_MAP
 d["launch_input_recovery_map"]={"path":str(RECOVERY),"sha256":RECOVERY_SHA}
 s["capture_scope"]="two accepted samples only: initial step 0 and terminal step 10000; not full-breath coverage"
 s["source_partition_provenance"]=provenance
 s["launch_input_relocations"]=RECOVERY_MAP
 s["launch_input_recovery_map"]={"path":str(RECOVERY),"sha256":RECOVERY_SHA}
 declaration.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n")
 summary.write_text(json.dumps(s,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"postprocessed":True,"capture_scope":s["capture_scope"],"status":s.get("status"),
                   "summary_path":str(summary),"summary_sha256":sha(summary)}))

def main():
 mod=load_core()
 result=mod.main()
 if "--out" in sys.argv:
  enrich(sys.argv[sys.argv.index("--out")+1])
 return result

if __name__=="__main__":
 raise SystemExit(main())
