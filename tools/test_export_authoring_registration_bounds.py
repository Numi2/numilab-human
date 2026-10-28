#!/usr/bin/env python3
from pathlib import Path
import numpy as np, importlib.util, time, json, hashlib, sys, argparse
HERE=Path(__file__).resolve().parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
sys.path.insert(0,str(HERE))
m=load("pruned230",HERE/"export_authoring_registered_surface_sidecars_230.py")
ap=argparse.ArgumentParser()
ap.add_argument("--native-parent",type=Path)
ap.add_argument("--native-outer",type=Path)
ap.add_argument("--legacy-exporter",type=Path)
ap.add_argument("--benchmark-output",type=Path)
args=ap.parse_args()
def exhaustive(parent,index,material,p,tie=1e-8):
 values=[]
 for r in np.flatnonzero(parent["materials"]==material):
  q,b,d=m.native.closest_point_on_parent_tet(parent,index,int(r),p);values.append((d,int(r),q,b))
 best=min(x[0] for x in values);ties=sorted([x for x in values if abs(x[0]-best)<=tie],key=lambda x:(x[0],x[1]))
 if max(np.linalg.norm(x[2]-ties[0][2]) for x in ties)>max(tie,1e-9):raise ValueError("distinct projections")
 return ties
def compare(a,b):
 assert a["tet"]==b["tet"] and a["ties"]==b["ties"],(a["tet"],b["tet"],a["ties"],b["ties"])
 for k in ("distance","projected","bary","residual"):
  assert np.array_equal(np.asarray(a[k]),np.asarray(b[k])),(k,a[k],b[k])
for_case=[]
points=[];tets=[];materials=[]
for j,scale in enumerate([1e-5,1e-4,.001,.005,.02,.1]):
 origin=np.array([j*.4,0.0,0.0]);tet=origin+scale*np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
 tets.append(np.arange(len(points),len(points)+4));points.extend(tet);materials.append(15)
 for_case.extend((15,p) for p in [tet.mean(0),tet[0],tet[[1,2,3]].mean(0),tet[0]+np.array([0,0,-scale*.1]),tet[[0,1]].mean(0)])
# A coincident additional tet must remain in the full current tie set.
tets.append(tets[2].copy());materials.append(15)
# One different-material near query must never leak into a material-15 result.
tets.append(tets[2].copy());materials.append(16)
par={"points":np.asarray(points),"tets":np.asarray(tets),"materials":np.asarray(materials),"donors":np.arange(8),"regions":np.arange(8)+10,"faces":np.arange(8)+100}
idx=m.native.build_index(par);mi=m.build_material_indices(par)
for material,p in for_case:
 got=m.closest_same_material(par,idx,mi,material,p,1,1e-8)
 ref=exhaustive(par,idx,material,p)
 assert got["tet"]==ref[0][1] and got["ties"]==[x[1] for x in ref]
 assert got["distance"]==ref[0][0] and np.array_equal(got["bary"],ref[0][3])

# Separate equally near projections remain a rejection, not a tie-ID shortcut.
p2={"points":np.array([[-1,0,0],[-2,0,0],[-1,1,0],[-1,0,-1],[1,0,0],[2,0,0],[1,0,-1],[1,1,0]],float),
 "tets":np.arange(8).reshape(2,4),"materials":np.array([15,15]),"donors":np.array([0,1]),"regions":np.array([0,1]),"faces":np.array([0,1])}
i2=m.native.build_index(p2);mi2=m.build_material_indices(p2)
rejected=False
try:m.closest_same_material(p2,i2,mi2,15,np.zeros(3),2,1e-8)
except ValueError as e:rejected="distinct projected" in str(e)
assert rejected
print(json.dumps({"exhaustive_cases":len(for_case),"coincident_ties_preserved":True,"distinct_projection_ties_rejected":True,"scalar_projection_policy_unchanged":True}),flush=True)
if not any((args.native_parent,args.native_outer,args.legacy_exporter,args.benchmark_output)):sys.exit(0)
if not all((args.native_parent,args.native_outer,args.legacy_exporter,args.benchmark_output)):
 ap.error("native benchmark requires parent, outer, legacy-exporter and benchmark-output together")
old=load("reference230",args.legacy_exporter)
out=args.benchmark_output;out.mkdir(exist_ok=False)
p,ix,mat,meta=m.load_parent_current(args.native_parent);legacy=old.build_material_indices(p)
z=np.load(args.native_outer);Q=np.asarray(z["points_world_m"],float)
chosen=np.unique(np.r_[np.linspace(0,len(Q)-1,18,dtype=int),11456,4202,32769,1952,22989,38299]);samples=Q[chosen]
timings=[];results={}
for pair in range(3):
 row={}
 for label in (("legacy","pruned") if pair%2==0 else ("pruned","legacy")):
  mod,ind=(old,legacy) if label=="legacy" else (m,mat)
  t=time.monotonic();r=[mod.closest_same_material(p,ix,ind,15,q,.02,1e-8) for q in samples];row[label+"_s"]=time.monotonic()-t
  if "reference" not in results:results["reference"]=r
  else:
   for a,b in zip(results["reference"],r):compare(a,b)
 timings.append(row);print(json.dumps({"pair":pair,**row}),flush=True)
r={"physical_admission":False,"simulation_steps":0,"scope":"CPU asset-preparation parent registration; not GPU solver throughput","source_sha256":hashlib.sha256((HERE/"export_authoring_registered_surface_sidecars_230.py").read_bytes()).hexdigest(),"legacy_source_sha256":hashlib.sha256(args.legacy_exporter.read_bytes()).hexdigest(),"parent":meta,"candidate_path":str(args.native_outer),"candidate_sha256":hashlib.sha256(args.native_outer.read_bytes()).hexdigest(),"sample_source_vertex_rows":chosen.tolist(),"matched_pairs":timings,"all_result_fields_bitwise_equal":True,"median_speedup":float(np.median([x["legacy_s"]/x["pruned_s"] for x in timings])),"synthetic_exhaustive_cases":len(for_case)}
(out/"receipt.json").write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps({"bitwise_equal":True,"samples":len(chosen),"median_speedup":r["median_speedup"]}),flush=True)
