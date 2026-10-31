"""Prepare a complete cumulative surface endpoint quotient, including its failures.
This is a diagnostic remeshing input, never physical admission. Desired boundaries
come from the complete current surface plus authenticated original layer walls.
"""
from pathlib import Path
import numpy as np,json,time,importlib.util,argparse
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("m209",H/"transfer_surface_contractions_to_volume_209_root.py")
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
B=H/"fixed-boundary-quality-topology-196"
BATCHES=("collapse-batch-002","collapse-batch-003-lineage-corrected-002","collapse-batch-004-from-003-001")
def face_keys(f):
 sf=np.sort(f,axis=1);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]))
 return sf,order
def parity(f):
 return 1-2*((((f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8)))%2)
def crosswalk(points,lookup):
 out=[]
 for p in points:
  k=m.key(p)
  if k not in lookup:raise ValueError("desired surface point missing from physical input")
  out.append(lookup[k])
 return np.asarray(out,np.int64)
def role_shell(role):
 f=H/("complete-typed-shell-owner-074/skin-shell.npz" if role=="Skin" else "fat-inferred-boundary-shell-assembly-140-v2/fat-shell-refined-typed.npz")
 with np.load(f) as z:a={k:z[k] for k in z.files}
 p=a["points_world_m"];faces=a["faces"];marker=a["face_marker"] if "face_marker" in a else a["face_markers"]
 return f,p,faces,marker,a
def target(v,role):
 f,sp,sf,marker,arrays=role_shell(role);lut=crosswalk(sp,v["look"])
 rawroles=("O","I") if role=="Skin" else ("I","D")
 targets=[];parts={}
 for rr in rawroles:
  reference=sf[marker==1] if rr=="O" else sf[arrays["registered_"+rr+"_shell_face_rows"]]
  reference=lut[reference];rkeys={tuple(sorted(map(int,x))):int(y) for x,y in zip(reference,parity(reference))}
  dp=B/"dp-batch-004"/("candidate-"+rr+"-fixed-boundary-dp-raw.npz")
  with np.load(dp) as z:pp=z["points_world_m"];ff=z["faces"]
  mm=crosswalk(pp,v["look"]);mapped=mm[ff];sg=parity(mapped)
  votes=[int(a)*rkeys[tuple(sorted(map(int,t)))] for t,a in zip(mapped,sg) if tuple(sorted(map(int,t))) in rkeys]
  if not votes or len(set(votes))!=1:raise ValueError("ambiguous role winding "+role+"/"+rr)
  flip=votes[0]<0
  current=B/BATCHES[-1]/("candidate-"+rr+"-short-edge-collapse-raw.npz")
  with np.load(current) as z:pp=z["points_world_m"];ff=z["faces"]
  mapped=crosswalk(pp,v["look"])[ff]
  if flip:mapped=mapped[:,[0,2,1]]
  targets.append(mapped);parts[rr]={"path":str(current),"sha256":m.sha(current),"faces":len(mapped),"winding_flip":flip}
 wallmarker=3 if role=="Skin" else 6
 wall=lut[sf[marker==wallmarker]];targets.append(wall)
 desired=np.vstack(targets)
 # Full shell is required: each edge exactly twice in opposite directions.
 ed=np.vstack([desired[:,[0,1]],desired[:,[1,2]],desired[:,[2,0]]])
 sorteded=np.sort(ed,axis=1);order=np.lexsort((sorteded[:,1],sorteded[:,0]));ee=sorteded[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ee[1:]!=ee[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ee)])
 signs=np.where(ed[:,0]<ed[:,1],1,-1);sm=np.add.reduceat(signs[order],starts)
 if np.any(ct!=2) or np.any(sm!=0):raise ValueError("desired boundary not oriented closed "+role)
 return desired,parts,{"path":str(f),"sha256":m.sha(f),"wall_faces":len(wall)}
def run(a):
 a.output.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 volumes={r:m.read_volume(r) for r in ("Skin","Fat")};ops=[];pins={}
 for batch in BATCHES:
  ledger=B/batch/"accepted-operations.jsonl";pins[str(ledger)]=m.sha(ledger)
  lines=[json.loads(l) for l in ledger.read_text().splitlines()]
  for oi,op in enumerate(lines):
   rr=op["role"]
   raw=B/batch/("candidate-"+rr+"-short-edge-collapse-raw.npz")
   # Current and parent point domains are the same immutable coordinate array.
   if rr not in pins:
    with np.load(raw) as z:coords=z["points_world_m"].astype(np.float32)
    pins[rr]=coords
   coords=pins[rr]
   keep=int(op["retained_input_point_row"]);drop=int(op["collapsed_input_point_row"])
   for vr in (("Skin",) if rr=="O" else ("Fat",) if rr=="D" else ("Skin","Fat")):
    v=volumes[vr];u=v["look"][m.key(coords[keep])];w=v["look"][m.key(coords[drop])]
    if v["map"][u]!=u or v["map"][w]!=w:raise ValueError("non-active endpoint in authenticated ledger")
    v["map"][w]=u
   ops.append({"batch":batch,"row":oi,"role":rr,"keep":keep,"drop":drop,"patch_id":op["patch_id"]})
 report=[]
 for vr,v in volumes.items():
  # Path compress all accumulated endpoint maps.
  mp=v["map"]
  while True:
   nxt=mp[mp]
   if np.array_equal(nxt,mp):break
   mp=nxt
  tt=mp[v["t"]];ss=np.sort(tt,axis=1);keep=~np.any(ss[:,1:]==ss[:,:-1],axis=1);tt=tt[keep]
  d,q=m.m.measurements(v["p"],tt);bb=m.boundary(tt)
  desired,parts,walls=target(v,vr)
  actualkeys={tuple(sorted(map(int,f))) for f in bb};desiredkeys={tuple(sorted(map(int,f))) for f in desired}
  missing=np.array([f for f in desired if tuple(sorted(map(int,f))) not in actualkeys],np.int64).reshape(-1,3)
  extra=np.array([f for f in bb if tuple(sorted(map(int,f))) not in desiredkeys],np.int64).reshape(-1,3)
  output=a.output/(vr.lower()+"-complete-surface-quotient-remesh-input.npz")
  np.savez_compressed(output,points_world_m=v["p"],tetrahedra=tt,material_indices=v["mat"][keep],desired_boundary_faces=desired,combinatorial_boundary_faces=bb,desired_missing_faces=missing,current_extra_boundary_faces=extra,input_node_quotient=mp,retained_input_cell_rows=np.flatnonzero(keep),nonpositive_cell_rows=np.flatnonzero(d<=0),quality_failure_cell_rows=np.flatnonzero(~m.m.passes(d,q)))
  report.append({"role":vr,"input":str(v["path"]),"input_sha256":m.sha(v["path"]),"output":str(output),"output_sha256":m.sha(output),"cells":len(tt),"removed_degenerate_cells":int(sum(~keep)),"nonpositive_cells":int(sum(d<=0)),"quality_failures":int(sum(~m.m.passes(d,q))),"desired_faces":len(desired),"desired_missing_faces":len(missing),"current_extra_boundary_faces":len(extra),"desired_boundary_closed_oriented":True,"source_parts":parts,"walls":walls})
 rr={"status":"COMPLETE_SURFACE_QUOTIENT_DIAGNOSTIC_REQUIRES_LOCAL_REMESH","physical_admission":False,"native_steps":0,"script_sha256":m.sha(Path(__file__)),"ledger_pins":{k:v for k,v in pins.items() if k not in ("O","I","D")},"operations":len(ops),"roles":report,"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(rr,indent=2)+"\n");print(json.dumps(rr,indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);run(ap.parse_args())
