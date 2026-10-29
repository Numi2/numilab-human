#!/usr/bin/env python3
"""Necessary capacity check on bounded anatomical mass-graph components.
Offline preparation only. Passing does not certify feasible flow or physics.
"""
import argparse,csv,json,math,time,importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("cap510",P/"audit_regional_mass_capacity_464.py");cap=importlib.util.module_from_spec(s);s.loader.exec_module(cap)
def components(parent_centroids,parent_mass,parent_keys,child_centroids,child_volume,child_keys,radius):
 arrays=[np.asarray(v) for v in [parent_centroids,parent_mass,parent_keys,child_centroids,child_volume,child_keys]]
 pc,pm,pk,cc,cv,ck=arrays
 if radius<=0 or not math.isfinite(radius):raise ValueError("invalid bounded graph radius")
 if pc.shape!=(len(pm),3) or cc.shape!=(len(cv),3) or pk.shape!=(len(pm),3) or ck.shape!=(len(cv),3):raise ValueError("incomplete component domain")
 if not all(np.isfinite(v).all() for v in arrays) or np.any(pm<=0) or np.any(cv<=0):raise ValueError("invalid mass graph geometry or supply")
 if not np.issubdtype(pk.dtype,np.integer) or not np.issubdtype(ck.dtype,np.integer) or np.any(pk<0) or np.any(ck<0):raise ValueError("mass graph requires typed nonnegative anatomical IDs")
 rows=[];edge_count=0
 for key in sorted(set(map(tuple,pk))|set(map(tuple,ck))):
  ps=np.flatnonzero(np.all(pk==key,axis=1));cs=np.flatnonzero(np.all(ck==key,axis=1))
  owner=np.arange(len(ps)+len(cs));edges=cKDTree(pc[ps]).query_ball_tree(cKDTree(cc[cs]),radius) if len(ps) and len(cs) else [[] for p in ps]
  def find(i):
   while owner[i]!=i:owner[i]=owner[owner[i]];i=owner[i]
   return int(i)
  for i,js in enumerate(edges):
   for j in js:
    d=pc[ps[i]]-cc[cs[j]];dist=float(np.sqrt(d@d))
    if dist>radius or float(d@d)>radius*radius*(1+1e-14):continue
    edge_count+=1;a=find(i);b=find(len(ps)+j)
    if a!=b:owner[b]=a
  groups={}
  for i,p in enumerate(ps):groups.setdefault(find(i),[[],[]])[0].append(int(p))
  for j,c in enumerate(cs):groups.setdefault(find(len(ps)+j),[[],[]])[1].append(int(c))
  if key[0] not in cap.DENSITY_BOUNDS:raise ValueError("unknown physical material")
  lo,hi=cap.DENSITY_BOUNDS[key[0]]
  for pa,ch in groups.values():
   mass=float(np.sum(pm[pa],dtype=np.longdouble));vol=float(np.sum(cv[ch],dtype=np.longdouble))
   rows.append({"material_donor_region":list(map(int,key)),"parent_rows":pa,"child_rows":ch,"parent_mass_kg":mass,"child_reference_volume_m3":vol,"allowed_child_volume_m3":[mass/hi,mass/lo],"component_capacity_feasible":bool(pa and ch and vol*lo<=mass<=vol*hi),"isolated_parent":bool(pa and not ch),"isolated_child":bool(ch and not pa)})
 return rows,edge_count
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--candidate",type=Path,required=True);ap.add_argument("--parent-prefix",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 def read(suffix):
  with Path(str(a.parent_prefix)+suffix).open() as f:return list(csv.DictReader(f))
 nr=read(".nodes.csv");tr=read(".tets.csv");lr=read(".labels.csv")
 if len(tr)!=len(lr):raise ValueError("source mass and label rows differ")
 if any(int(r["parent_node_index"])!=i for i,r in enumerate(nr)):raise ValueError("source node identity order mismatch")
 for i,(r,l) in enumerate(zip(tr,lr)):
  if int(r["parent_tet_row"])!=i or int(l["parent_tet_row"])!=i:raise ValueError("source tet identity order mismatch")
  if int(r["material_index"])!=int(l["material_index"]) or int(r["donor_body"])!=int(l["geometric_donor_body"]):raise ValueError("source material/donor identity mismatch")
 pp=np.array([[float(r["rest_"+k+"_m"]) for k in "xyz"] for r in nr]);pt=np.array([[int(r["parent_node"+str(k)]) for k in range(4)] for r in tr]);pm=np.array([float(r["parent_mass_kg"]) for r in tr])
 pk=np.array([[int(r[k]) for k in ["material_index","geometric_donor_body","anatomical_region_id"]] for r in lr]);pc=pp[pt].mean(1)
 with np.load(a.candidate) as z:
  x=z["points_reference_world_m_f64"].astype(np.float32).astype(float);t=z["volume_tetrahedra"].astype(int);cp=z["child_parent_tet_row"].astype(int)
  ck=np.stack([z[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]],axis=1)
  if not np.array_equal(ck,pk[cp]):raise ValueError("declared source labels changed")
 if t.ndim!=2 or t.shape[1]!=4 or len(t)!=len(ck) or t.size==0 or t.min()<0 or t.max()>=len(x):raise ValueError("invalid candidate tetrahedron domain")
 cv=cap.tet_volumes_gradients(x,t)[0];cc=x[t].mean(1);d=np.linalg.norm(cc-pc[cp],axis=1);radius=max(.0002,math.ceil(float(d.max())/1e-6)*1e-6+1e-6)
 report,edges=components(pc,pm,pk,cc,cv,ck,radius)
 bad=[r for r in report if not r["component_capacity_feasible"]]
 receipt={"status":"bounded_mass_component_capacity_diagnostic","scope":"necessary component capacity only; no Hall subset proof, native flow admission, geometry or physics claim","candidate_sha256":cap.sha(a.candidate),"script_sha256":cap.sha(__file__),"parent_sha256":{suffix:cap.sha(Path(str(a.parent_prefix)+suffix)) for suffix in [".nodes.csv",".tets.csv",".labels.csv"]},"coordinate_contract":"canonical Float32 child-reference world metres; authoritative original parent rest coordinates","maximum_declared_source_centroid_distance_m":float(d.max()),"predeclared_distance_envelope_m":radius,"distance_rule":"ceil maximum declared source-parent centroid displacement to1um +1um, floor200um","component_count":len(report),"failed_components":len(bad),"bounded_bipartite_edges":edges,"density_bounds_kg_m3":cap.DENSITY_BOUNDS,"components":report,"physical_admission":False,"elapsed_s":time.monotonic()-start}
 (a.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
 print(json.dumps({k:v for k,v in receipt.items() if k not in ["components","parent_sha256"]}),flush=True)
if __name__=="__main__":main()
