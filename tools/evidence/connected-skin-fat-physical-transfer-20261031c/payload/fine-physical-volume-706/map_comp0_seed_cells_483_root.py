#!/usr/bin/env python3
"""Rebase frozen comp0 cavity seed cells by ordered tetrahedron and material identity."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
PARENT=ROOT/"joint-D414-I466-target-475-root/fat-target397/fat-current-typed-target-rebound.npz"
CURRENT=ROOT/"fixed-fat-coordinate-views-496-root/fat-all-coordinate-views-volume-proposal.npz"
SHELL=ROOT/"fat-comp0-connected-warning-repair-412/run-025-current-target-475-shell/connected-cavity-shells.npz"
MAP=ROOT/"fat-comp0-connected-warning-repair-412/run-025-current-target-475-shell/base-to-current-cell-map.npz"
SHELL_REPORT=ROOT/"fat-comp0-connected-warning-repair-412/run-025-current-target-475-shell/report.json"
PINS={"parent_candidate":"c1903e60b03aeb3a53592c8ecab020abae59eb8829899dece614d4a946f6fb04","current_candidate":"99cb88d89d29abfe0afff7219ad2ea0a56edbc36b98250c6103f8e1b6bce82fb","shell":"1e2054ff6d12971977f25e9460ba5a87820ddd39dc0fe12afd364a733e527f93","row_map":"34b1c8695d883fff1a5397c9887cddde3f36a14d7fe2574363e8b153147bd94"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key_array(t,mat):
 dtype=np.dtype([("material","<i4"),("v0","<i8"),("v1","<i8"),("v2","<i8"),("v3","<i8")])
 out=np.empty(len(t),dtype=dtype);out["material"]=mat
 for j in range(4):out[f"v{j}"]=t[:,j]
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 for name,path in (("parent_candidate",PARENT),("current_candidate",CURRENT),("shell",SHELL),("row_map",MAP)):
  got=sha(path)
  if PINS[name] and got!=PINS[name]:raise ValueError(f"{name} SHA mismatch: {got}")
 report0=json.loads(SHELL_REPORT.read_text())
 if report0["inputs"]["current_volume"]["path"]!=str(PARENT) or report0["inputs"]["current_volume"]["sha256"]!=PINS["parent_candidate"]:raise ValueError("run025 report parent mismatch")
 with np.load(SHELL,allow_pickle=False) as z:rows=np.asarray(z["input_cell_rows"],np.int64)
 with np.load(MAP,allow_pickle=False) as z:expanded=np.asarray(z["expanded_selected_cell_rows"],np.int64)
 if len(rows)!=13499 or len(expanded)!=len(rows) or not np.array_equal(np.sort(rows),np.sort(expanded)):raise ValueError("run025 shell and expanded-row map disagree")
 with np.load(PARENT,allow_pickle=False) as z:
  parent_t=np.asarray(z["tetrahedra"],np.int64);parent_m=np.asarray(z["material_indices"],np.int32);parent_p=np.asarray(z["points_world_m"],np.float32)
 with np.load(CURRENT,allow_pickle=False) as z:
  current_t=np.asarray(z["tetrahedra"],np.int64);current_m=np.asarray(z["material_indices"],np.int32);current_p=np.asarray(z["points_world_m"],np.float32)
 if rows.min()<0 or rows.max()>=len(parent_t):raise ValueError("run025 rows exceed parent tetrahedron domain")
 source_t=parent_t[rows];source_m=parent_m[rows]
 if parent_p.shape!=current_p.shape:raise ValueError("point index domains differ; no implicit remap permitted")
 used=np.unique(source_t)
 if used.min()<0 or used.max()>=len(parent_p):raise ValueError("selected cell point row outside parent")
 if not np.array_equal(parent_p[used].view(np.uint32),current_p[used].view(np.uint32)):raise ValueError("a frozen comp0 seed point changed in the 496 candidate")
 current_keys=key_array(current_t,current_m);order=np.argsort(current_keys,order=("material","v0","v1","v2","v3"),kind="stable");sorted_keys=current_keys[order]
 if np.any(sorted_keys[1:]==sorted_keys[:-1]):raise ValueError("current candidate has duplicate ordered tet/material keys")
 query=key_array(source_t,source_m);pos=np.searchsorted(sorted_keys,query);safe=np.minimum(pos,len(sorted_keys)-1)
 if np.any(pos>=len(sorted_keys)) or not np.array_equal(sorted_keys[safe],query):
  miss=np.flatnonzero((pos>=len(sorted_keys))|(sorted_keys[safe]!=query));raise ValueError(f"{len(miss)} frozen comp0 tets absent; examples {miss[:12].tolist()}")
 mapped=np.asarray(order[pos],np.int64)
 if len(np.unique(mapped))!=len(mapped):raise ValueError("comp0 seed maps non-injectively")
 if not np.array_equal(current_t[mapped],source_t) or not np.array_equal(current_m[mapped],source_m):raise ValueError("ordered tet/material readback failed")
 a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir()
 seed=a.output/"comp0-seed-cell-rows.npz"
 np.savez_compressed(seed,current_candidate_cell_rows=mapped,parent_c190_cell_rows=rows,parent_tetrahedra=source_t,material_indices=source_m,source_used_point_rows=used)
 receipt={"schema":"numi.fat.comp0-exact-seed-cell-rebase.v1","status":"PASS_EXACT_ORDERED_TET_MATERIAL_SEED_REBASE","physical_admission":False,"native_steps":0,
  "parent_candidate":{"path":str(PARENT),"sha256":sha(PARENT)},"current_candidate":{"path":str(CURRENT),"sha256":sha(CURRENT)},
  "run025_shell":{"path":str(SHELL),"sha256":sha(SHELL)},"run025_row_map":{"path":str(MAP),"sha256":sha(MAP)},"run025_report":{"path":str(SHELL_REPORT),"sha256":sha(SHELL_REPORT)},
  "seed_rows":{"path":str(seed),"sha256":sha(seed),"count":len(mapped),"array_sha256":hashlib.sha256(mapped.tobytes()).hexdigest()},
  "identity_checks":{"run025_shell_rows_equal_expanded_map":True,"ordered_tetrahedron_and_material_exact_match":True,"point_index_domain_identical":True,"selected_point_f32_bytes_unchanged":True,"mapped_rows_injective":True},
  "selected_cells":len(mapped),"selected_points":len(used),"limitations":"This is a row-identity seed map only. The target boundary is recomputed separately from the current typed surfaces; no geometry, remesh, mass, native, or physical admission is claimed."}
 out=a.output/"receipt.json";out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n");print(json.dumps(receipt,indent=2,sort_keys=True))
if __name__=="__main__":main()
