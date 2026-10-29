#!/usr/bin/env python3
"""Cheap regional capacity witnesses for canonical-Float32 physical candidates.
This diagnoses the unchanged native material/donor/region density capacities.
It neither redistributes mass nor admits geometry or sparse transport.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

DENSITY_BOUNDS={15:(900.,1100.),16:(850.,950.)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def tet_volumes_gradients(points,tets):
 a,b,c,d=(points[tets[:,k]] for k in range(4))
 ba,ca,da=b-a,c-a,d-a
 gb=np.cross(ca,da)/6.;gc=np.cross(da,ba)/6.;gd=np.cross(ba,ca)/6.
 gradients=np.stack((-(gb+gc+gd),gb,gc,gd),axis=1)
 return np.einsum("ij,ij->i",ba,gb),gradients

def capacity_rows(points,tets,keys,parent_supply):
 volume,gradient=tet_volumes_gradients(points,tets)
 rows=[];node_rows=[];grad_rows=[];grad_groups=[]
 child_keys={tuple(map(int,k)) for k in keys}
 for key in sorted(set(parent_supply)|child_keys):
  selected=np.flatnonzero(np.all(keys==np.asarray(key),axis=1))
  supply=float(parent_supply.get(key,0.))
  lower,upper=DENSITY_BOUNDS[key[0]]
  v=float(volume[selected].sum());minimum=lower*v;maximum=upper*v
  nodes=np.unique(tets[selected]);acc=np.zeros((len(points),3))
  np.add.at(acc,tets[selected].ravel(),gradient[selected].reshape(-1,3))
  nz=nodes[np.any(acc[nodes]!=0.,axis=1)]
  ix=len(rows);node_rows.extend(nz.tolist());grad_rows.extend(acc[nz].tolist());grad_groups.extend([ix]*len(nz))
  rows.append({"material_donor_region":list(key),"parent_supply_kg":supply,
   "candidate_cells":int(len(selected)),"candidate_reference_volume_m3":v,
   "allowed_reference_volume_m3":[supply/upper,supply/lower],
   "minimum_mass_capacity_kg":minimum,"maximum_mass_capacity_kg":maximum,
   "mass_capacity_feasible":bool(minimum<=supply<=maximum),
   "volume_deficit_to_nearest_bound_m3":max(supply/upper-v,0.)+min(supply/lower-v,0.),
   "volume_gradient_node_count":len(nz)})
 return rows,volume,{"group_row":np.asarray(grad_groups,np.int64),
  "volume_node_row":np.asarray(node_rows,np.int64),
  "d_group_volume_d_world_position_m2":np.asarray(grad_rows,float).reshape(-1,3)}

def load_supply(tet_file,label_file):
 with Path(tet_file).open() as f:mass=list(csv.DictReader(f))
 with Path(label_file).open() as f:labels=list(csv.DictReader(f))
 if len(mass)!=len(labels):raise ValueError("parent mass/label row counts differ")
 groups={}
 for i,(m,l) in enumerate(zip(mass,labels)):
  if int(m["parent_tet_row"])!=i or int(l["parent_tet_row"])!=i:raise ValueError("parent row order mismatch")
  if int(m["material_index"])!=int(l["material_index"]) or int(m["donor_body"])!=int(l["geometric_donor_body"]):raise ValueError("parent mass/label identity mismatch")
  key=tuple(int(l[k]) for k in ["material_index","geometric_donor_body","anatomical_region_id"])
  if key[0] not in DENSITY_BOUNDS:raise ValueError("unhandled physical material")
  groups.setdefault(key,[]).append(float(m["parent_mass_kg"]))
 return {k:float(np.sum(np.asarray(v,np.longdouble),dtype=np.longdouble)) for k,v in groups.items()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--parent-tets",required=True);ap.add_argument("--parent-labels",required=True);ap.add_argument("--out",required=True)
 args=ap.parse_args();out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
 with np.load(args.candidate,allow_pickle=False) as z:
  p=np.asarray(z["points_reference_world_m_f64"],np.float32).astype(float)
  t=np.asarray(z["volume_tetrahedra"],np.int64)
  keys=np.stack([z[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]],axis=1).astype(np.int64)
 if not np.isfinite(p).all() or t.ndim!=2 or t.shape[1]!=4 or t.min()<0 or t.max()>=len(p):raise ValueError("invalid candidate domain")
 rows,vol,grad=capacity_rows(p,t,keys,load_supply(args.parent_tets,args.parent_labels))
 np.savez_compressed(out/"group-volume-gradients.npz",**grad)
 failed=[r for r in rows if not r["mass_capacity_feasible"]]
 receipt={"status":"regional_mass_capacity_diagnostic","scope":"necessary aggregate capacity only; not sufficient for distance-limited sparse mass transport, source provenance, geometric validity or physical admission",
  "candidate_sha256":sha(args.candidate),"parent_tets_sha256":sha(args.parent_tets),
  "parent_labels_sha256":sha(args.parent_labels),"script_sha256":sha(__file__),
  "canonical_coordinates":"Float32 reference world metres, same field as native mass importer",
  "density_bounds_kg_m3":{str(k):list(v) for k,v in DENSITY_BOUNDS.items()},
  "group_count":len(rows),"failed_group_count":len(failed),"nonpositive_cell_count":int(np.count_nonzero(vol<=0)),
  "minimum_signed_cell_volume_m3":float(vol.min()),"physical_admission":False,
  "groups":rows,"failed_groups":failed,"gradient_sha256":sha(out/"group-volume-gradients.npz")}
 (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
 print(json.dumps({k:receipt[k] for k in ["status","group_count","failed_group_count","nonpositive_cell_count","minimum_signed_cell_volume_m3"]}))
if __name__=="__main__":main()
