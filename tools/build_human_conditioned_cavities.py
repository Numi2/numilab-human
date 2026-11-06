#!/usr/bin/env python3
"""Compile fixed-coordinate Human cavities for the existing native admission owner."""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,itertools,json,time
import numpy as np

def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def module(path,name,expected):
 if sha(path)!=expected:raise ValueError("cavity owner SHA mismatch: "+str(path))
 spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out

def spectral_condition(points,cells):
 """Permutation-, scale- and translation-invariant centered tetrahedral aspect."""
 x=np.asarray(points,np.float64)[np.asarray(cells)]
 if x.ndim!=3 or x.shape[1:]!=(4,3) or not np.isfinite(x).all():raise ValueError("invalid finite tetrahedral geometry")
 s=np.linalg.svd(x-x.mean(axis=1,keepdims=True),compute_uv=False)
 return np.divide(s[:,0],s[:,-1],out=np.full(len(x),np.inf),where=s[:,-1]>0)

def cavity_payload(source_sha,cell_count,patches):
 retired=np.concatenate([np.asarray(p["old_rows"],np.uint32) for p in patches])
 if len(np.unique(retired))!=len(retired) or retired.max()>=cell_count:raise ValueError("overlapping cavity rows")
 keep=np.ones(cell_count,bool);keep[retired]=False
 new=np.concatenate([np.asarray(p["new_cells"],np.uint32) for p in patches])
 offsets=np.r_[0,np.cumsum([len(p["new_cells"]) for p in patches])].astype(np.uint32)
 parent_chunks=[np.tile(np.asarray(p["old_rows"],np.uint32),len(p["new_cells"])) for p in patches]
 parent_counts=np.concatenate([np.full(len(p["new_cells"]),len(p["old_rows"]),np.uint32) for p in patches])
 return dict(retained_input_cell_rows=np.flatnonzero(keep).astype(np.uint32),
  replacement_tetrahedra=new,replacement_tet_offsets=offsets,
  replacement_parent_offsets=np.r_[0,np.cumsum(parent_counts)].astype(np.uint32),
  replacement_parent_cell_rows=np.concatenate(parent_chunks),source_volume_sha256=np.array(source_sha))

def validate_mass_proof(parity, receipt, volume_sha, mass_sha, receipt_sha, cell_count):
 # Use the admitted allocator's existing relative tolerance plus its roundoff allowance.
 if (parity["status"]!="PASS_INDEPENDENT_RECORDED_FLOW_TARGET_PARITY" or
     parity["target"]["sha256"]!=mass_sha or parity["mass_receipt"]["sha256"]!=receipt_sha or
     parity["prescribed_cells"]!=cell_count or
     not np.isfinite(parity["maximum_relative_mass_difference"]) or
     parity["maximum_relative_mass_difference"]>1e-10+64*np.finfo(float).eps):
  raise ValueError("mass target is not independently verified")
 if (receipt["status"]!="PASS_SOURCE_ONLY_REPLAY" or
     receipt["candidate"]["mass_transport_envelope"]["input_sha256"]["candidate"]!=volume_sha or
     receipt["candidate"]["child_tetrahedron_count"]!=cell_count):
  raise ValueError("source mass receipt is not bound to this volume")

def run(a):
 start=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 for path,digest in [(a.volume,a.volume_sha256),(a.reference,a.reference_sha256),(a.mass_csv,a.mass_sha256),(a.mass_parity,a.mass_parity_sha256),(a.mass_receipt,a.mass_receipt_sha256)]:
  if sha(path)!=digest:raise ValueError("input SHA mismatch: "+str(path))
 if sha(a.polygon_owner.parent/"fixed_boundary_volume_flip_198_root.py")!=a.predicate_sha256:raise ValueError("exact predicate SHA mismatch")
 polygon=module(a.polygon_owner,"human_polygon",a.polygon_sha256)
 mesh=module(Path(__file__).with_name("simplify_human_skin_fat_boundary.py"),"human_quotient",a.mesh_owner_sha256)
 with np.load(a.volume,allow_pickle=False) as z:
  p=z["points_reference_world_m_f64"].astype(np.float32);t=z["volume_tetrahedra"].astype(np.int64)
  labels=np.column_stack([z[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]])
  if not np.array_equal(p.astype(np.float64),z["points_reference_world_m_f64"]):raise ValueError("volume is not exact emitted Float32")
 mr=np.loadtxt(a.mass_csv,delimiter=",",skiprows=1,ndmin=2)
 if mr.shape!=(len(t),2) or not np.array_equal(mr[:,0],np.arange(len(t))) or not np.all(np.isfinite(mr)) or np.any(mr[:,1]<=0):raise ValueError("invalid prescribed mass domain")
 mass=mr[:,1].copy();del mr
 parity=json.loads(a.mass_parity.read_text())
 source_receipt=json.loads(a.mass_receipt.read_text())
 validate_mass_proof(parity,source_receipt,a.volume_sha256,a.mass_sha256,a.mass_receipt_sha256,len(t))
 u,receipt=mesh.load_reference_deformation(a.reference,a.reference_sha256,p,t,labels)
 gate=mesh.ReferenceDeformationGate(p,t,u,labels[:,1],receipt)
 cond=np.empty(len(t))
 for lo in range(0,len(t),8192):cond[lo:lo+8192]=spectral_condition(p,t[lo:lo+8192])
 if not np.all(np.isfinite(cond)):raise ValueError("nonfinite input shape condition")
 seeds=np.flatnonzero(cond>=a.condition_threshold)
 seeds=seeds[np.lexsort((seeds,-cond[seeds]))][:a.maximum_seeds]
 inc=mesh.incidence(t,len(p))
 def edge_rows(d,e):return np.intersect1d(inc(int(d)),inc(int(e)),assume_unique=True)
 reject=collections.Counter();seen=set();used=set();patches=[]
 for seed in seeds:
  if time.monotonic()-start>a.seconds or len(patches)>=a.maximum_cavities:break
  for d,e in itertools.combinations(map(int,t[seed]),2):
   key=tuple(sorted((d,e)))
   if key in seen:continue
   seen.add(key);rows=edge_rows(d,e)
   if len(rows) not in (3,4):reject["bounded_degree"]+=1;continue
   if not np.all(labels[rows]==labels[rows[0]]):reject["material_donor_region_boundary"]+=1;continue
   old=t[rows];nodes=set(map(int,old.ravel()))
   if nodes&used:reject["dependent_cavity"]+=1;continue
   ring=polygon.ring_order(old,d,e)
   if ring is None:reject["open_edge_link"]+=1;continue
   proposal=polygon.best_triangulation(p,ring,d,e)
   if proposal is None:reject["no_positive_triangulation"]+=1;continue
   new,_,tris=proposal
   od,oq=polygon.m.measurements(p,old);nd,nq=polygon.m.measurements(p,new)
   if len(new)>len(old) or np.any(nd/6<=1e-18) or np.any(nq<1e-5) or float(nq.min())<1.02*float(oq.min()):reject["unchanged_native_quality_gate"]+=1;continue
   oldc=cond[rows];newc=spectral_condition(p,new)
   if newc.max()>oldc.max() or np.log(np.maximum(newc/1000,1)).sum()>=np.log(np.maximum(oldc/1000,1)).sum()-.02:reject["spectral_no_progress"]+=1;continue
   perimeter={tuple(sorted((ring[i],ring[(i+1)%len(ring)]))) for i in range(len(ring))}
   diagonals={tuple(sorted(edge)) for tri in tris for edge in itertools.combinations(tri,2)}-perimeter
   if any(len(edge_rows(*edge)) for edge in diagonals):reject["existing_new_diagonal"]+=1;continue
   if any(len(np.intersect1d(edge_rows(v0,v1),inc(v2),assume_unique=True)) for v0,v1,v2 in tris):reject["existing_separator_face"]+=1;continue
   ok,why=polygon.m.certify(p,old,new)
   if not ok:reject["exact_"+why]+=1;continue
   density=float(mass[rows].sum()/(od.sum()/6));bounds={15:(900.,1100.),16:(850.,950.)}.get(int(labels[rows[0],0]))
   if bounds is None or density<bounds[0]-64*np.finfo(float).eps*bounds[1] or density>bounds[1]+64*np.finfo(float).eps*bounds[1]:reject["density"]+=1;continue
   padded=old.copy();padded[:len(new)]=new;live=np.arange(len(old))<len(new)
   deformation=gate.propose(rows,padded,live)
   if deformation is None:reject["reference_deformation"]+=1;continue
   gate.commit(rows,live,deformation);used.update(nodes)
   patches.append(dict(old_rows=rows.tolist(),new_cells=new.tolist(),removed_edge=[d,e],ring=ring,
    material=int(labels[rows[0],0]),donor=int(labels[rows[0],1]),region=int(labels[rows[0],2]),
    minimum_native_quality_before=float(oq.min()),minimum_native_quality_after=float(nq.min()),
    maximum_spectral_condition_before=float(oldc.max()),maximum_spectral_condition_after=float(newc.max()),
    density_kg_m3=density,exact_boundary_and_volume=True))
   if len(patches)%32==0:print("passing_cavities",len(patches),"elapsed_s",time.monotonic()-start,flush=True)
   break
 report=dict(status="PASS_FIXED_BOUNDARY_CONDITIONING_PROPOSAL" if patches else "NO_ADMISSIBLE_CONDITIONING_PROGRESS",
  input=dict(volume=dict(path=str(a.volume),sha256=a.volume_sha256),reference=dict(path=str(a.reference),sha256=a.reference_sha256),mass_csv=dict(path=str(a.mass_csv),sha256=a.mass_sha256),mass_parity=dict(path=str(a.mass_parity),sha256=a.mass_parity_sha256),mass_receipt=dict(path=str(a.mass_receipt),sha256=a.mass_receipt_sha256)),
  owners={str(path):sha(path) for path in [Path(__file__),a.polygon_owner,a.polygon_owner.parent/"fixed_boundary_volume_flip_198_root.py",Path(mesh.__file__)]},
  input_cells=len(t),eligible_seed_cells=len(seeds),tested_edges=len(seen),passing_cavities=len(patches),rejections=dict(reject),patches=patches,
  reference_deformation=gate.report(),source_coordinate_and_typed_boundaries_unchanged=True,
  source_mass_transport_and_native_admission="MUST_REPEAT_EXISTING_NATIVE_OWNER",physical_admission=False,native_steps=0,elapsed_s=time.monotonic()-start)
 if patches:
  output=a.output/"fixed-boundary-volume-proposal.npz";np.savez_compressed(output,**cavity_payload(a.volume_sha256,len(t),patches))
  report["output"]=dict(path=str(output),sha256=sha(output))
 (a.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k not in ["patches","reference_deformation"]},indent=2),flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ["volume","reference","mass-csv","mass-parity","mass-receipt","polygon-owner","output"]:ap.add_argument("--"+name,type=Path,required=True)
 for name in ["volume","reference","mass","mass-parity","mass-receipt","polygon","predicate","mesh-owner"]:ap.add_argument("--"+name+"-sha256",required=True)
 ap.add_argument("--seconds",type=float,default=180);ap.add_argument("--maximum-seeds",type=int,default=4096);ap.add_argument("--maximum-cavities",type=int,default=256);ap.add_argument("--condition-threshold",type=float,default=10000)
 a=ap.parse_args()
 if a.seconds<=0 or a.maximum_seeds<1 or a.maximum_cavities<1 or a.condition_threshold<=1:raise ValueError("invalid bounded construction settings")
 run(a)
if __name__=="__main__":main()
