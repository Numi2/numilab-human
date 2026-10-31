"""Batch fixed-coordinate flips for exact native-apex boundary obstructions.
Reuses existing surface quality, typed patch-union, full-context exact owners.
Output is an unadmitted raw topology proposal; matching volumes remain mandatory.
"""
from pathlib import Path
import numpy as np,json,hashlib,argparse,sys,importlib.util,itertools,time,collections
H=Path(__file__).resolve().parent;S=H/"fixed-boundary-quality-topology-196"
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
base=module("base197for259",S/"run_fixed_boundary_dp_batch_197.py")
joint=module("joint272for273",H/"audit_joint_native_apex_272_root.py")
def sha(p):return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def cyc(f):
 a,b,c=map(int,f);return min((a,b,c),(b,c,a),(c,a,b))
def run(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 census=H/"joint-native-apex-272-root/current-census-001/report.json";cr=json.loads(census.read_text())
 inputs={r:Path(v["path"]) for r,v in zip(("O","I","D"),cr["surfaces"])}
 arrays={r:base.load(p) for r,p in inputs.items()}
 for r,v in zip(("O","I","D"),cr["surfaces"]):assert sha(inputs[r])==v["sha256"]
 pts={r:np.asarray(v["points_world_m"],np.float32) for r,v in arrays.items()};faces={r:np.asarray(v["faces"],np.int64) for r,v in arrays.items()}
 pred=base.module("exact042for259",base.OWNER).load_exact_owner()
 assert sha(Path(pred.__file__))==base.OWNER_IMPL_SHA
 bone=base.module("bone609for259",base.BONE);filtermod=base.module("filter030for259",base.FILTER)
 pbase={};chunks=[];cursor=0
 for r in ("D","O","I"):pbase[r]=cursor;chunks.append(pts[r]);cursor+=len(pts[r])
 combined=np.concatenate(chunks);records=[];facebase=base.FACE_BASE
 for r in ("D","O","I"):
  raw,_=pred.float32_surface_records(pts[r].astype(float),faces[r],source_seam_groups=())
  assert len(raw)==len(faces[r])
  records.extend((z[0],z[1],z[2],facebase[r]+i,tuple(pbase[r]+int(v) for v in z[4])) for i,z in enumerate(raw))
 lookup={}
 for r in pts:
  for i,p in enumerate(pts[r]):
   key=p.tobytes();assert key not in lookup;lookup[key]=(r,i)
 for vol,marker,roles,wbase in (("Skin",3,("O","I"),900000),("Fat",6,("I","D"),901000)):
  path,pin=base.SHELL[vol];assert sha(path)==pin;z=base.load(path);sf=z["faces"][z["face_marker"]==marker];mp={}
  for i in np.unique(sf):
   r,j=lookup[np.asarray(z["points_world_m"][i],np.float32).tobytes()];assert r in roles;mp[int(i)]=pbase[r]+j
  mapped=np.asarray([[mp[int(v)] for v in f] for f in sf],np.int64)
  raw,_=pred.float32_surface_records(combined.astype(float),mapped,source_seam_groups=());assert len(raw)==90
  records.extend((z[0],z[1],z[2],wbase+i,tuple(map(int,z[4]))) for i,z in enumerate(raw))
 bones,bone_receipt=bone.load_bone_barrier_records(pred,root=H.parent,vertex_offset=len(combined),expected_native_sha256=base.BONE_NATIVE,expected_manifest_sha256=base.BONE_MANIFEST)
 assert len(bones)==938272
 index=pred._prepare_surface_aabb(records+bones)
 allowed={facebase[r]+i for r in faces for i in range(len(faces[r]))}
 filtered=filtermod.FilteredPreparedExactOwner(pred,allowed_retirable_face_ids=allowed,excluded_face_ids=())
 accepted=[];rejected=[];retired={r:set() for r in faces};used={r:set() for r in faces};accepted_records=[];newid=4000000
 for role in ("O","I"):
  P=pts[role];F=faces[role];q=joint.score(P,F)
  edgeindex=collections.defaultdict(list)
  for row,f in enumerate(F):
   for i in range(3):edgeindex[tuple(sorted((int(f[i]),int(f[(i+1)%3]))))].append(row)
  candidates=[]
  for seed in np.flatnonzero(q<a.surface_floor):
   for i in range(3):
    edge=tuple(sorted((int(F[seed,i]),int(F[seed,(i+1)%3]))));rows=edgeindex[edge]
    if len(rows)!=2:continue
    rows=tuple(sorted(rows));old=F[list(rows)];verts=set(map(int,old.ravel()))
    if len(verts)!=4:continue
    fa=old[0];k=next(j for j in range(3) if {int(fa[j]),int(fa[(j+1)%3])}==set(edge))
    u,v,w=map(int,(fa[k],fa[(k+1)%3],fa[(k+2)%3]));d=next(iter(set(map(int,old[1]))-set(edge)))
    if edgeindex.get(tuple(sorted((w,d)))):continue
    new=np.asarray([[w,d,v],[d,w,u]],np.int64);nq=joint.score(P,new)
    if np.any(nq<a.surface_floor):continue
    oldxyz=P[old].astype(float);newxyz=P[new].astype(float)
    nold=np.cross(oldxyz[:,1]-oldxyz[:,0],oldxyz[:,2]-oldxyz[:,0]).sum(axis=0)
    nnew=np.cross(newxyz[:,1]-newxyz[:,0],newxyz[:,2]-newxyz[:,0])
    if np.any(nnew@nold<=0):continue
    # Exact combinatorial oriented boundary proof for the two triangles.
    def chain(fs):
     C=collections.Counter()
     for aa,bb,cc in fs:
      for xx,yy in ((aa,bb),(bb,cc),(cc,aa)):C[tuple(sorted((int(xx),int(yy))))]+=1 if xx<yy else -1
     return {k:v for k,v in C.items() if v}
    if chain(old)!=chain(new):continue
    candidates.append((int(np.count_nonzero(q[list(rows)]<a.surface_floor)),-float(nq.min()),rows,new,verts))
  seen=set()
  for _,__,rows,new,verts in sorted(candidates,key=lambda v:(-v[0],v[1],v[2])):
   if rows in seen:continue
   seen.add(rows)
   if verts&used[role]:continue
   excluded={facebase[r]+i for r in faces for i in retired[r]};excluded.update(facebase[role]+i for i in rows);filtered.set_excluded(excluded)
   raw,_=pred.float32_surface_records(P.astype(float),new,source_seam_groups=())
   recs=[(z[0],z[1],z[2],newid+j,tuple(pbase[role]+int(v) for v in z[4])) for j,z in enumerate(raw)]
   external=filtered._audit_pair_prepared_first(index,recs,same_surface=True)
   inside=pred._audit_record_pairs(itertools.chain(itertools.combinations(recs,2),itertools.product(recs,accepted_records)),same_surface=True)
   if external["count"] or inside["count"]:
    rejected.append({"role":role,"rows":list(rows),"external_pairs":external["count"],"internal_pairs":inside["count"]});continue
   union=base.union_tuples(arrays[role],rows)
   old=F[list(rows)];xyz=P.astype(float)
   flux=lambda ff:float(np.einsum("ij,ij->i",xyz[ff[:,0]],np.cross(xyz[ff[:,1]],xyz[ff[:,2]])).sum()/6)
   op={"role":role,"parent_face_rows":list(rows),"new_faces":new.tolist(),"typed_parent_union":union,"typed_parent_union_sha256":base.union_digest(union),"source_face_ancestry_union":base.ancestry_union(arrays[role],rows),"q_before":q[list(rows)].tolist(),"q_after":joint.score(P,new).tolist(),"oriented_surface_flux_delta_m3":flux(new)-flux(old),"exact_changed_vs_complete_pairs":0,"exact_new_internal_and_previous_pairs":0,"vertices_unchanged":True}
   accepted.append(op);accepted_records.extend(recs);newid+=2;retired[role].update(rows);used[role].update(verts)
 filtered.set_excluded({facebase[r]+i for r in faces for i in retired[r]})
 joint=filtered._audit_pair_prepared_first(index,accepted_records,same_surface=True)
 jointself=pred._audit_record_pairs(itertools.combinations(accepted_records,2),same_surface=True)
 assert joint["count"]==jointself["count"]==0
 outputs={}
 for role in ("O","I"):
  F=faces[role];G=F.copy();ops=[x for x in accepted if x["role"]==role]
  changed=[];newunions=[];offset=[0];patchid=[]
  fmap=np.arange(len(F),dtype=np.int64)
  for i,op in enumerate(ops):
   rows=np.asarray(op["parent_face_rows"],np.int64);G[rows]=np.asarray(op["new_faces"]);fmap[rows]=-1;changed.extend(rows.tolist())
  changed=np.asarray(sorted(changed),np.int64)
  for row in changed:
   op=next(op for op in ops if int(row) in op["parent_face_rows"]);newunions.extend(op["typed_parent_union"]);offset.append(len(newunions));patchid.append("native-apex-floor259-%s-%d"%(role,ops.index(op)))
  bd0,top0=base.topology(F);bd1,top1=base.topology(G)
  assert np.array_equal(bd0,bd1) and top0==top1
  assert len(np.unique(np.sort(G,axis=1),axis=0))==len(G)
  U=np.asarray(newunions,np.int64).reshape(-1,7)
  path=a.output/("candidate-%s-native-apex-floor-raw.npz"%role)
  np.savez_compressed(path,points_world_m=arrays[role]["points_world_m"],points_world_m_f32=pts[role],faces=G,point_output_to_parent_rows=np.arange(len(pts[role]),dtype=np.int64),face_output_to_parent_rows=fmap,retired_parent_face_rows=changed,new_faces_parent_point_rows=G[changed],changed_parent_face_rows=changed,new_face_patch_union_offsets=np.asarray(offset,np.int64),new_output_face_patch_id=np.asarray(patchid),immediate_parent_candidate_sha256=np.asarray([sha(inputs[role])]),**{base.UNION[k].replace("quality_star_patch_union_face_source_","new_face_patch_union_"):U[:,k] for k in range(7)})
  q0=joint.score(pts[role],F);q1=joint.score(pts[role],G)
  outputs[role]={"path":str(path),"sha256":sha(path),"parent":str(inputs[role]),"parent_sha256":sha(inputs[role]),"bad_floor_before":int(sum(q0<a.surface_floor)),"bad_floor_after":int(sum(q1<a.surface_floor)),"no_native_apex_before":int(sum(q0<joint.GATE)),"no_native_apex_after":int(sum(q1<joint.GATE)),"changed_face_rows":changed.tolist()}
 report={"status":"PASS_EXACT_CONNECTED_OI_APEX_FLOOR_TOPOLOGY_PROPOSAL","physical_admission":False,"native_steps":0,"source_sha256":sha(Path(__file__)),"census_sha256":sha(census),"construction_surface_floor":a.surface_floor,"quality_metric":"joint_volume_shape_apex_upper_bound","joint_bound_owner_sha256":sha(Path(joint.__file__)),"native_gate_unchanged":True,"all_point_arrays_unchanged":True,"joint_complete_surface_walls_all185_pairs":joint["count"],"joint_changed_self_pairs":jointself["count"],"accepted_operations":accepted,"rejected_candidates":rejected,"outputs":outputs,"elapsed_s":time.monotonic()-tic,"typed_provenance":"RAW_GEOMETRY_PLUS_COMPLETE_PATCH_UNION_REQUIRES_EXISTING_TYPED_OWNER_REBIND","volume_transfer":"PENDING_MATCHING_SKIN_AND_FAT_CAVITY_COMPILATION"}
 (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k not in ("accepted_operations","rejected_candidates")},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--surface-floor",type=float,default=1.1e-5);a=ap.parse_args();run(a)
