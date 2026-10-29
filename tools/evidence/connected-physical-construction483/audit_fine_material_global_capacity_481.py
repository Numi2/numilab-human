#!/usr/bin/env python3
"""Pinned fine skin/fat global capacity and nesting inventory; no mass assignment."""
from pathlib import Path
import argparse,importlib.util,csv,json,time
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
B=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union"
sp=importlib.util.spec_from_file_location("gate427",P/"audit_coarse_material_bone_volume_427.py");g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--closed-D",type=Path,default=P/"geometry-worker/matched-nesting-390/run-038-run037-fat-occupancy/closed-inputs/D-closed-run037-f32.npz");a=ap.parse_args()
 out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
 op=B/"minimum-thickness-minkowski-328/attempt-006-anchored-local-topology/run-030-global-derived-patch-remesh/attempt-002/connected-witness-repair-001/result-003/candidate-O-open-f32-typed.npz"
 cp=B/"minimum-thickness-minkowski-328/attempt-006-anchored-local-topology/run-024-local-quality-condition/candidate-O-closed-f64-geometry.npz"
 ip=P/"geometry-worker/matched-nesting-390/run-038-run037-fat-occupancy/closed-inputs/I-closed-current-f32.npz"
 parent=P/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.tets.csv"
 with np.load(op) as z:ov=z["points_world_m"].astype(np.float32).astype(float);of=z["faces"];mapping=z["closed_emitted_vertex_rows"]
 with np.load(cp) as z:
  pv=z["points_world_m_f64"].astype(np.float32).astype(float);cf=z["faces_f64"][z["temporary_cap_descendant_face_mask"]]
  inv=np.full(len(pv),-1,np.int64);inv[mapping]=np.arange(len(mapping));caps=inv[cf]
  assert np.all(caps>=0) and np.array_equal(ov[caps],pv[cf])
 solids={"O":g.solid(ov,np.r_[of,caps])}
 for name,path in [("I",ip),("D",a.closed_D)]:
  with np.load(path) as z:
   points=z["points_world_m_f64"] if "points_world_m_f64" in z else z["points_world_m"]
   faces=z["faces_f64"] if "faces_f64" in z else z["faces"]
   solids[name]=g.solid(points,faces)
 if any(s.status()!=g.mf.Error.NoError for s in solids.values()):raise ValueError("invalid closed solid")
 with parent.open() as f:
  massrows=list(csv.DictReader(f))
 layers={}
 for material,name,outer,inner,lo,hi in [(15,"skin","O","I",900.,1100.),(16,"fat","I","D",850.,950.)]:
  domain=solids[outer]-solids[inner];volume=float(domain.volume());mass=sum(float(r["parent_mass_kg"]) for r in massrows if int(r["material_index"])==material)
  layers[name]={"material_index":material,"volume_m3":volume,"parent_mass_kg":mass,"fixed_density_bounds_kg_m3":[lo,hi],"implied_mean_density_kg_m3":mass/volume,"allowed_total_volume_m3":[mass/hi,mass/lo],"necessary_global_capacity_feasible":lo*volume<=mass<=hi*volume}
 r={"status":"pinned_fine_material_global_capacity_and_nesting_inventory","inputs":{k:{"path":str(v),"sha256":g.sha(v)} for k,v in [("O",op),("O_cap_parent",cp),("I",ip),("D",a.closed_D),("parent_mass",parent)]},"script_sha256":g.sha(__file__),"explicit_head_caps":len(caps),"head_vertices_float32_bit_identical":True,"solids":{n:{"volume_m3":float(s.volume()),"status":str(s.status())} for n,s in solids.items()},"layers":layers,"nesting_residual_m3":{"I_outside_O":float((solids["I"]-solids["O"]).volume()),"D_outside_I":float((solids["D"]-solids["I"]).volume())},"physical_admission":False,"scope":"necessary global mass capacity and raw solid nesting only; original open head loops remain separately identified; source group mass/embedding/physiology/native motion not admitted"}
 (out/"receipt.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r))
if __name__=="__main__":main()
